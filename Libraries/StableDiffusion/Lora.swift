// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN

/// LoRA weight structure for Stable Diffusion models.
///
/// LoRA (Low-Rank Adaptation) weights consist of two matrices:
/// - `up`: shape [out_dim, rank] - the upward projection
/// - `down`: shape [rank, in_dim] - the downward projection
/// - `alpha`: optional scaling factor for the adaptation
struct LoraWeight {
    let up: MLXArray      // shape: [out_dim, rank]
    let down: MLXArray    // shape: [rank, in_dim]
    let alpha: Float?     // optional scaling factor

    /// Compute the effective LoRA delta matrix.
    ///
    /// For linear layers: `up @ down` scaled by `alpha / rank`.
    /// The result is added to the original weight to fuse the LoRA.
    func delta(scale: Float = 1.0) -> MLXArray {
        let rank = Float(down.dim(0))
        let alphaScale = (alpha ?? rank) / rank
        return matmul(up, down) * MLXArray(alphaScale * scale)
    }
}

/// Mutable accumulator used while parsing LoRA safetensors files.
///
/// Collects `up`, `down`, and `alpha` tensors keyed by their base parameter name
/// before assembling them into immutable ``LoraWeight`` values.
private class LoraWeightAccumulator {
    var up: MLXArray?
    var down: MLXArray?
    var alpha: Float?
}

/// A container for all LoRA weights in a model.
struct LoraWeights {
    /// Mapping from parameter name to LoRA weights
    let weights: [String: LoraWeight]

    init(weights: [String: LoraWeight]) {
        self.weights = weights
    }
}

/// Load LoRA weights from a .safetensors file.
///
/// Handles common LoRA key formats:
/// - `lora_unet_*` / `lora_te_*` (A1111 / Kohya style)
/// - `unet.*` / `text_encoder.*` (diffusers style)
/// - Keys ending in `.lora_down.weight`, `.lora_up.weight`, `.alpha`
///
/// - Parameter url: Path to the .safetensors file containing LoRA weights
/// - Returns: A `LoraWeights` container with all loaded LoRA weights
func loadLoraWeights(url: URL) throws -> LoraWeights {
    let rawWeights = try loadArrays(url: url)

    // Phase 1: Accumulate up/down/alpha by base key
    var accumulators: [String: LoraWeightAccumulator] = [:]

    for (key, value) in rawWeights {
        // Skip metadata
        if key == "__metadata__" { continue }

        // Determine the type (up, down, or alpha) and extract the base key
        let baseKey: String
        let kind: String

        if key.hasSuffix(".lora_down.weight") {
            baseKey = String(key.dropLast(".lora_down.weight".count))
            kind = "down"
        } else if key.hasSuffix(".lora_up.weight") {
            baseKey = String(key.dropLast(".lora_up.weight".count))
            kind = "up"
        } else if key.hasSuffix(".alpha") {
            baseKey = String(key.dropLast(".alpha".count))
            kind = "alpha"
        } else {
            // Unknown key format, skip
            continue
        }

        let acc = accumulators[baseKey] ?? LoraWeightAccumulator()
        accumulators[baseKey] = acc

        switch kind {
        case "down":
            acc.down = value.asType(.float32)
        case "up":
            acc.up = value.asType(.float32)
        case "alpha":
            if let scalar: Float = try? value.asType(.float32).item() {
                acc.alpha = scalar
            }
        default:
            break
        }
    }

    // Phase 2: Convert complete accumulators to LoraWeight values
    var loraWeights: [String: LoraWeight] = [:]
    for (baseKey, acc) in accumulators {
        guard let up = acc.up, let down = acc.down else {
            continue
        }

        // Reshape conv2d LoRA weights (4D) to 2D for matrix multiplication
        let finalUp: MLXArray
        let finalDown: MLXArray
        if up.ndim == 4 {
            // Conv2d LoRA: [out, rank, kH, kW] → [out, rank * kH * kW]
            let shape = up.shape
            finalUp = up.reshaped(shape[0], -1)
        } else {
            finalUp = up
        }
        if down.ndim == 4 {
            let shape = down.shape
            finalDown = down.reshaped(shape[0], -1)
        } else {
            finalDown = down
        }

        loraWeights[baseKey] = LoraWeight(up: finalUp, down: finalDown, alpha: acc.alpha)
    }

    return LoraWeights(weights: loraWeights)
}

// MARK: - LoRA Key Mapping

/// Map LoRA key names from Kohya/A1111 format to diffusers/MLX parameter paths.
///
/// Kohya format uses underscores and `lora_unet_` / `lora_te_` prefixes:
///   `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q`
///
/// Diffusers/HF format uses dots:
///   `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`
private func mapLoraKeyToParameterPath(_ loraKey: String) -> (component: String, path: String)? {
    var key = loraKey

    // Determine which component this LoRA targets
    let component: String
    if key.hasPrefix("lora_unet_") {
        component = "unet"
        key = String(key.dropFirst("lora_unet_".count))
    } else if key.hasPrefix("lora_te1_") || key.hasPrefix("lora_te_") {
        component = "text_encoder"
        key = key.hasPrefix("lora_te1_")
            ? String(key.dropFirst("lora_te1_".count))
            : String(key.dropFirst("lora_te_".count))
    } else if key.hasPrefix("lora_te2_") {
        component = "text_encoder_2"
        key = String(key.dropFirst("lora_te2_".count))
    } else if key.hasPrefix("unet.") {
        component = "unet"
        key = String(key.dropFirst("unet.".count))
    } else if key.hasPrefix("text_encoder.") {
        component = "text_encoder"
        key = String(key.dropFirst("text_encoder.".count))
    } else if key.hasPrefix("text_encoder_2.") {
        component = "text_encoder_2"
        key = String(key.dropFirst("text_encoder_2.".count))
    } else {
        // Try to infer from key content
        if key.contains("down_block") || key.contains("up_block") || key.contains("mid_block") {
            component = "unet"
        } else if key.contains("text_model") {
            component = "text_encoder"
        } else {
            component = "unet"  // default
        }
    }

    // Convert Kohya underscore-separated format to dot-separated
    // Pattern: blocks_0_attentions_1 → blocks.0.attentions.1
    // We need to be careful: some parts like "to_q", "to_k" use underscores legitimately
    var dotPath = kohyaToDotPath(key)

    // Apply the same key remapping used for loading HF weights
    if component == "unet" {
        for rule in unetRules {
            dotPath = rule(dotPath) ?? dotPath
        }
        // Handle ff.net.0 split (linear1/linear2)
        if dotPath.contains("ff.net.0") {
            dotPath = dotPath.replacingOccurrences(of: "ff.net.0.proj", with: "linear1")
        }
    } else if component.hasPrefix("text_encoder") {
        for rule in clipRules {
            dotPath = rule(dotPath) ?? dotPath
        }
    }

    return (component, dotPath)
}

/// Convert Kohya underscore-separated key to dot-separated path.
///
/// E.g. `down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q`
/// → `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q`
private func kohyaToDotPath(_ key: String) -> String {
    // If already dot-separated, return as-is
    if key.contains(".") { return key }

    // Split on underscore and reconstruct with dots where appropriate
    let parts = key.split(separator: "_")
    var result: [String] = []
    var i = 0

    while i < parts.count {
        let part = String(parts[i])

        // If this part is a number, it's an index — append with dot
        if Int(part) != nil {
            result.append(part)
            i += 1
            continue
        }

        // Known compound names that use underscores
        let compoundNames = [
            "down_blocks", "up_blocks", "mid_blocks", "mid_block",
            "transformer_blocks", "time_emb_proj", "conv_shortcut",
            "block_out_channels", "conv_in", "conv_out", "conv_norm_out",
            "time_embedding", "add_embedding",
            "to_q", "to_k", "to_v", "to_out",
            "proj_in", "proj_out",
            "group_norm", "layer_norm1", "layer_norm2",
            "final_layer_norm", "text_model", "text_projection",
            "token_embedding", "position_embedding",
            "self_attn", "out_proj", "ff_net",
        ]

        // Try to match compound name
        var matched = false
        for compound in compoundNames {
            let compoundParts = compound.split(separator: "_")
            if i + compoundParts.count <= parts.count {
                let candidate = parts[i..<(i + compoundParts.count)].joined(separator: "_")
                if candidate == compound {
                    result.append(compound)
                    i += compoundParts.count
                    matched = true
                    break
                }
            }
        }

        if !matched {
            result.append(part)
            i += 1
        }
    }

    return result.joined(separator: ".")
}

// MARK: - LoRA Fusion

/// Fuse LoRA weights into a model by iterating over the LoRA keys
/// and looking up matching parameters by exact path.
///
/// - Parameters:
///   - model: The model to fuse LoRA weights into
///   - loraWeights: The LoRA weights container (keys are already mapped model paths)
///   - scale: Global scale factor for all LoRA weights (default 1.0)
private func fuseLora(model: Module, loraWeights: LoraWeights, scale: Float = 1.0) {
    // Build a lookup from param path → param value
    let params = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })

    var updates: [(String, MLXArray)] = []
    var matched = 0

    for (loraPath, loraWeight) in loraWeights.weights {
        // The LoRA path targets a ".weight" parameter inside a module.
        // Try exact match first, then with ".weight" suffix
        let paramKey: String
        let paramValue: MLXArray

        if let v = params[loraPath + ".weight"] {
            paramKey = loraPath + ".weight"
            paramValue = v
        } else if let v = params[loraPath] {
            paramKey = loraPath
            paramValue = v
        } else {
            // No match found for this LoRA key
            continue
        }

        // Compute the LoRA delta
        var delta = loraWeight.delta(scale: scale)

        // Handle shape mismatch between LoRA delta and parameter
        if delta.shape != paramValue.shape {
            if paramValue.ndim == 4 {
                // Conv2d weights in MLX: [outC, kH, kW, inC]
                let shape = paramValue.shape
                if delta.shape[0] == shape[0] && delta.count == paramValue.count {
                    delta = delta.reshaped(shape)
                } else {
                    print("  LoRA skip (shape mismatch): \(loraPath) delta=\(delta.shape) param=\(paramValue.shape)")
                    continue
                }
            } else if paramValue.ndim == 2 && delta.ndim == 2 {
                if delta.shape[0] == paramValue.shape[1] && delta.shape[1] == paramValue.shape[0] {
                    delta = delta.transposed()
                } else {
                    print("  LoRA skip (shape mismatch): \(loraPath) delta=\(delta.shape) param=\(paramValue.shape)")
                    continue
                }
            } else {
                print("  LoRA skip (ndim mismatch): \(loraPath) delta=\(delta.shape) param=\(paramValue.shape)")
                continue
            }
        }

        let newValue = paramValue + delta.asType(paramValue.dtype)
        updates.append((paramKey, newValue))
        matched += 1
    }

    if !updates.isEmpty {
        _ = try? model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
        eval(model)
    }

    print("  → Matched \(matched)/\(loraWeights.weights.count) LoRA weights")
}

/// Load and fuse LoRA weights from a .safetensors file into a StableDiffusion model.
///
/// This is the main entry point for applying LoRA to a model.
///
/// - Parameters:
///   - sd: The StableDiffusion model to fuse LoRA into
///   - loraUrl: Path to the .safetensors file containing LoRA weights
///   - scale: Scale factor for LoRA weights (default 1.0)
public func loadAndFuseLora(_ sd: StableDiffusion, loraUrl: URL, scale: Float = 1.0) throws {
    let loraWeights = try loadLoraWeights(url: loraUrl)

    // Separate weights by component and map keys to model parameter paths
    var unetWeights: [String: LoraWeight] = [:]
    var te1Weights: [String: LoraWeight] = [:]
    var te2Weights: [String: LoraWeight] = [:]

    for (key, weight) in loraWeights.weights {
        if let mapped = mapLoraKeyToParameterPath(key) {
            switch mapped.component {
            case "unet":
                unetWeights[mapped.path] = weight
            case "text_encoder":
                te1Weights[mapped.path] = weight
            case "text_encoder_2":
                te2Weights[mapped.path] = weight
            default:
                break
            }
        }
    }

    // Fuse into UNet
    if !unetWeights.isEmpty {
        print("Fusing \(unetWeights.count) LoRA weights into UNet...")
        fuseLora(model: sd.unet, loraWeights: LoraWeights(weights: unetWeights), scale: scale)
    }

    // Fuse into text encoder 1
    if !te1Weights.isEmpty {
        print("Fusing \(te1Weights.count) LoRA weights into text encoder...")
        fuseLora(
            model: sd.textEncoder, loraWeights: LoraWeights(weights: te1Weights), scale: scale)
    }

    // Fuse into text encoder 2 (SDXL only)
    if let sdxl = sd as? StableDiffusionXL, !te2Weights.isEmpty {
        print("Fusing \(te2Weights.count) LoRA weights into text encoder 2...")
        fuseLora(
            model: sdxl.textEncoder2, loraWeights: LoraWeights(weights: te2Weights), scale: scale)
    }
}
