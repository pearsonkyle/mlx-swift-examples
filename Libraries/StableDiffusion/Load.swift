// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/model_io.py

/// Configuration for loading stable diffusion weights.
///
/// These options can be tuned to conserve memory.
public struct LoadConfiguration: Sendable {

    /// convert weights to float16
    public var float16 = true

    /// quantize weights
    public var quantize = false

    public var dType: DType {
        float16 ? .float16 : .float32
    }

    public init(float16: Bool = true, quantize: Bool = false) {
        self.float16 = float16
        self.quantize = quantize
    }
}

/// Parameters for evaluating a stable diffusion prompt and generating latents
public struct EvaluateParameters: Sendable {

    /// `cfg` value from the preset
    public var cfgWeight: Float

    /// number of steps -- default is from the preset
    public var steps: Int

    /// number of images to generate at a time
    public var imageCount = 1
    public var decodingBatchSize = 1

    /// size of the latent tensor -- the result image is a factor of 8 larger than this
    public var latentSize = [64, 64]

    public var seed: UInt64
    public var prompt = ""
    public var negativePrompt = ""

    public init(
        cfgWeight: Float, steps: Int, imageCount: Int = 1, decodingBatchSize: Int = 1,
        latentSize: [Int] = [64, 64], seed: UInt64? = nil, prompt: String = "",
        negativePrompt: String = ""
    ) {
        self.cfgWeight = cfgWeight
        self.steps = steps
        self.imageCount = imageCount
        self.decodingBatchSize = decodingBatchSize
        self.latentSize = latentSize
        self.seed = seed ?? UInt64(Date.timeIntervalSinceReferenceDate * 1000)
        self.prompt = prompt
        self.negativePrompt = negativePrompt
    }
}

/// File types for ``StableDiffusionConfiguration/files``. Used by the presets to provide
/// relative file paths for different types of files.
enum FileKey {
    case unetConfig
    case unetWeights
    case textEncoderConfig
    case textEncoderWeights
    case textEncoderConfig2
    case textEncoderWeights2
    case vaeConfig
    case vaeWeights
    case diffusionConfig
    case tokenizerVocabulary
    case tokenizerMerges
    case tokenizerVocabulary2
    case tokenizerMerges2
}

/// Stable diffusion configuration -- this selects the model to load.
///
/// Use the preset values:
/// - ``presetSDXLTurbo``
/// - ``presetStableDiffusion21Base``
///
/// or use the enum (convenient for command line tools):
///
/// - ``Preset/sdxlTurbo``
/// - ``Preset/sdxlTurbo``
///
/// Call ``download(hub:progressHandler:)`` to download the weights, then
/// ``textToImageGenerator(hub:configuration:)`` or
/// ``imageToImageGenerator(hub:configuration:)`` to produce the ``ImageGenerator``.
///
/// The ``ImageGenerator`` has a method to generate the latents:
/// - ``TextToImageGenerator/generateLatents(parameters:)``
/// - ``ImageToImageGenerator/generateLatents(image:parameters:strength:)``
///
/// Evaluate each of the latents from that iterator and use the decoder to turn the last latent
/// into an image:
///
/// - ``ImageGenerator/decode(xt:)``
///
/// Finally use ``Image`` to save it to a file or convert to a CGImage for display.
public struct StableDiffusionConfiguration: Sendable {
    public let id: String
    let files: [FileKey: String]
    public let defaultParameters: @Sendable () -> EvaluateParameters
    let factory:
        @Sendable (HubApi, StableDiffusionConfiguration, LoadConfiguration) throws ->
            StableDiffusion

    public func download(
        hub: HubApi = HubApi(), progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws {
        let repo = Hub.Repo(id: self.id)
        try await hub.snapshot(
            from: repo, matching: Array(files.values), progressHandler: progressHandler)
    }

    public func textToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
        throws -> TextToImageGenerator?
    {
        try factory(hub, self, configuration) as? TextToImageGenerator
    }

    public func imageToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
        throws -> ImageToImageGenerator?
    {
        try factory(hub, self, configuration) as? ImageToImageGenerator
    }

    public enum Preset: String, Codable, CaseIterable, Sendable {
        case base
        case sdxlTurbo = "sdxl-turbo"

        public var configuration: StableDiffusionConfiguration {
            switch self {
            case .base: presetStableDiffusion21Base
            case .sdxlTurbo: presetSDXLTurbo
            }
        }
    }

    /// See https://huggingface.co/stabilityai/sdxl-turbo for the model details and license
    public static let presetSDXLTurbo = StableDiffusionConfiguration(
        id: "stabilityai/sdxl-turbo",
        files: [
            .unetConfig: "unet/config.json",
            .unetWeights: "unet/diffusion_pytorch_model.safetensors",
            .textEncoderConfig: "text_encoder/config.json",
            .textEncoderWeights: "text_encoder/model.safetensors",
            .textEncoderConfig2: "text_encoder_2/config.json",
            .textEncoderWeights2: "text_encoder_2/model.safetensors",
            .vaeConfig: "vae/config.json",
            .vaeWeights: "vae/diffusion_pytorch_model.safetensors",
            .diffusionConfig: "scheduler/scheduler_config.json",
            .tokenizerVocabulary: "tokenizer/vocab.json",
            .tokenizerMerges: "tokenizer/merges.txt",
            .tokenizerVocabulary2: "tokenizer_2/vocab.json",
            .tokenizerMerges2: "tokenizer_2/merges.txt",
        ],
        defaultParameters: { EvaluateParameters(cfgWeight: 0, steps: 2) },
        factory: { hub, sdConfiguration, loadConfiguration in
            let sd = try StableDiffusionXL(
                hub: hub, configuration: sdConfiguration, dType: loadConfiguration.dType)
            if loadConfiguration.quantize {
                quantize(model: sd.textEncoder, filter: { k, m in m is Linear })
                quantize(model: sd.textEncoder2, filter: { k, m in m is Linear })
                quantize(model: sd.unet, groupSize: 32, bits: 8)
            }
            return sd
        }
    )

    /// See https://huggingface.co/stabilityai/stable-diffusion-2-1-base for the model details and license
    public static let presetStableDiffusion21Base = StableDiffusionConfiguration(
        id: "stabilityai/stable-diffusion-2-1-base",
        files: [
            .unetConfig: "unet/config.json",
            .unetWeights: "unet/diffusion_pytorch_model.safetensors",
            .textEncoderConfig: "text_encoder/config.json",
            .textEncoderWeights: "text_encoder/model.safetensors",
            .vaeConfig: "vae/config.json",
            .vaeWeights: "vae/diffusion_pytorch_model.safetensors",
            .diffusionConfig: "scheduler/scheduler_config.json",
            .tokenizerVocabulary: "tokenizer/vocab.json",
            .tokenizerMerges: "tokenizer/merges.txt",
        ],
        defaultParameters: { EvaluateParameters(cfgWeight: 7.5, steps: 50) },
        factory: { hub, sdConfiguration, loadConfiguration in
            let sd = try StableDiffusionBase(
                hub: hub, configuration: sdConfiguration, dType: loadConfiguration.dType)
            if loadConfiguration.quantize {
                quantize(model: sd.textEncoder, filter: { k, m in m is Linear })
                quantize(model: sd.unet, groupSize: 32, bits: 8)
            }
            return sd
        }
    )

}

// MARK: - Key Mapping

func keyReplace(_ replace: String, _ with: String) -> @Sendable (String) -> String? {
    return { [replace, with] key in
        if key.contains(replace) {
            return key.replacingOccurrences(of: replace, with: with)
        }
        return nil
    }
}

func dropPrefix(_ prefix: String) -> @Sendable (String) -> String? {
    return { [prefix] key in
        if key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }
}

// see map_unet_weights()

let unetRules: [@Sendable (String) -> String?] = [
    // Map up/downsampling
    keyReplace("downsamplers.0.conv", "downsample"),
    keyReplace("upsamplers.0.conv", "upsample"),

    // Map the mid block
    keyReplace("mid_block.resnets.0", "mid_blocks.0"),
    keyReplace("mid_block.attentions.0", "mid_blocks.1"),
    keyReplace("mid_block.resnets.1", "mid_blocks.2"),

    // Map attention layers
    keyReplace("to_k", "key_proj"),
    keyReplace("to_out.0", "out_proj"),
    keyReplace("to_q", "query_proj"),
    keyReplace("to_v", "value_proj"),

    // Map transformer ffn
    keyReplace("ff.net.2", "linear3"),
]

func unetRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key
    var value = value

    for rule in unetRules {
        key = rule(key) ?? key
    }

    // Map transformer ffn
    if key.contains("ff.net.0") {
        let k1 = key.replacingOccurrences(of: "ff.net.0.proj", with: "linear1")
        let k2 = key.replacingOccurrences(of: "ff.net.0.proj", with: "linear2")
        let (v1, v2) = value.split()
        return [(k1, v1), (k2, v2)]
    }

    if key.contains("conv_shortcut.weight") {
        value = value.squeezed()
    }

    // Transform the weights from 1x1 convs to linear
    if value.ndim == 4 && (key.contains("proj_in") || key.contains("proj_out")) {
        value = value.squeezed()
    }

    if value.ndim == 4 {
        value = value.transposed(0, 2, 3, 1)
        value = value.reshaped(-1).reshaped(value.shape)
    }

    return [(key, value)]
}

let clipRules: [@Sendable (String) -> String?] = [
    dropPrefix("text_model."),
    dropPrefix("embeddings."),
    dropPrefix("encoder."),

    // Map attention layers
    keyReplace("self_attn.", "attention."),
    keyReplace("q_proj.", "query_proj."),
    keyReplace("k_proj.", "key_proj."),
    keyReplace("v_proj.", "value_proj."),

    // Map ffn layers
    keyReplace("mlp.fc1", "linear1"),
    keyReplace("mlp.fc2", "linear2"),
]

func clipRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key

    for rule in clipRules {
        key = rule(key) ?? key
    }

    // not used
    if key == "position_ids" {
        return []
    }

    // text_projection is a Linear module in CLIPTextModel, so the raw weight
    // tensor needs to be mapped to text_projection.weight
    if key == "text_projection" && value.ndim == 2 {
        return [("text_projection.weight", value)]
    }

    return [(key, value)]
}

let vaeRules: [@Sendable (String) -> String?] = [
    // Map up/downsampling
    keyReplace("downsamplers.0.conv", "downsample"),
    keyReplace("upsamplers.0.conv", "upsample"),

    // Map attention layers
    keyReplace("to_k", "key_proj"),
    keyReplace("to_out.0", "out_proj"),
    keyReplace("to_q", "query_proj"),
    keyReplace("to_v", "value_proj"),

    // Map the mid block
    keyReplace("mid_block.resnets.0", "mid_blocks.0"),
    keyReplace("mid_block.attentions.0", "mid_blocks.1"),
    keyReplace("mid_block.resnets.1", "mid_blocks.2"),

    keyReplace("mid_blocks.1.key.", "mid_blocks.1.key_proj."),
    keyReplace("mid_blocks.1.query.", "mid_blocks.1.query_proj."),
    keyReplace("mid_blocks.1.value.", "mid_blocks.1.value_proj."),
    keyReplace("mid_blocks.1.proj_attn.", "mid_blocks.1.out_proj."),

]

func vaeRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key
    var value = value

    for rule in vaeRules {
        key = rule(key) ?? key
    }

    // Map the quant/post_quant layers
    if key.contains("quant_conv") {
        key = key.replacingOccurrences(of: "quant_conv", with: "quant_proj")
        value = value.squeezed()
    }

    // Map the conv_shortcut to linear
    if key.contains("conv_shortcut.weight") {
        value = value.squeezed()
    }

    if value.ndim == 4 {
        value = value.transposed(0, 2, 3, 1)
        value = value.reshaped(-1).reshaped(value.shape)
    }

    return [(key, value)]
}

func loadWeights(
    url: URL, model: Module, mapper: (String, MLXArray) -> [(String, MLXArray)], dType: DType
) throws {
    let weights = try loadArrays(url: url).flatMap { mapper($0.key, $0.value.asType(dType)) }

    // Note: not using verifier because some shapes change upon load
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
}

// MARK: - Loading

func resolve(hub: HubApi, configuration: StableDiffusionConfiguration, key: FileKey) -> URL {
    precondition(
        configuration.files[key] != nil, "configuration \(configuration.id) missing key: \(key)")
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    return directory.appending(component: configuration.files[key]!)
}

func loadConfiguration<T: Decodable>(
    hub: HubApi, configuration: StableDiffusionConfiguration, key: FileKey, type: T.Type
) throws -> T {
    let url = resolve(hub: hub, configuration: configuration, key: key)
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
}

func loadUnet(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws
    -> UNetModel
{
    let unetConfiguration = try loadConfiguration(
        hub: hub, configuration: configuration, key: .unetConfig, type: UNetConfiguration.self)
    let model = UNetModel(configuration: unetConfiguration)

    let weightsURL = resolve(hub: hub, configuration: configuration, key: .unetWeights)
    try loadWeights(url: weightsURL, model: model, mapper: unetRemap, dType: dType)

    return model
}

func loadTextEncoder(
    hub: HubApi, configuration: StableDiffusionConfiguration,
    configKey: FileKey = .textEncoderConfig, weightsKey: FileKey = .textEncoderWeights, dType: DType
) throws -> CLIPTextModel {
    let clipConfiguration = try loadConfiguration(
        hub: hub, configuration: configuration, key: configKey,
        type: CLIPTextModelConfiguration.self)
    let model = CLIPTextModel(configuration: clipConfiguration)

    let weightsURL = resolve(hub: hub, configuration: configuration, key: weightsKey)
    try loadWeights(url: weightsURL, model: model, mapper: clipRemap, dType: dType)

    return model
}

func loadAutoEncoder(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws
    -> Autoencoder
{
    let autoEncoderConfiguration = try loadConfiguration(
        hub: hub, configuration: configuration, key: .vaeConfig, type: AutoencoderConfiguration.self
    )
    let model = Autoencoder(configuration: autoEncoderConfiguration)

    let weightsURL = resolve(hub: hub, configuration: configuration, key: .vaeWeights)
    try loadWeights(url: weightsURL, model: model, mapper: vaeRemap, dType: dType)

    return model
}

func loadDiffusionConfiguration(hub: HubApi, configuration: StableDiffusionConfiguration) throws
    -> DiffusionConfiguration
{
    try loadConfiguration(
        hub: hub, configuration: configuration, key: .diffusionConfig,
        type: DiffusionConfiguration.self)
}

// MARK: - Single File Loading

/// Local model paths for loading from local .safetensors files
public struct LocalModelPaths: Sendable {
    /// Path to the checkpoint file (.safetensors)
    public let checkpoint: URL
    /// Optional path to VAE file (.safetensors)
    public let vae: URL?
    /// Optional paths to LoRA files (.safetensors)
    public let loras: [(url: URL, scale: Float)]

    /// Create local model paths
    /// - Parameters:
    ///   - checkpoint: Path to the main checkpoint file
    ///   - vae: Optional path to VAE file
    ///   - lora: Optional path to a single LoRA file
    public init(checkpoint: URL, vae: URL? = nil, lora: URL? = nil) {
        self.checkpoint = checkpoint
        self.vae = vae
        self.loras = lora.map { [($0, Float(1.0))] } ?? []
    }

    /// Create local model paths with multiple LoRAs
    /// - Parameters:
    ///   - checkpoint: Path to the main checkpoint file
    ///   - vae: Optional path to VAE file
    ///   - loras: Array of (url, scale) tuples for LoRA weights
    public init(checkpoint: URL, vae: URL? = nil, loras: [(url: URL, scale: Float)]) {
        self.checkpoint = checkpoint
        self.vae = vae
        self.loras = loras
    }
}

// MARK: - LDM to Diffusers Key Conversion

/// Convert a UNet weight key from Stability AI's LDM format to HuggingFace diffusers format.
///
/// LDM format uses:
/// - `input_blocks.X.Y.*` → `down_blocks`/`conv_in`
/// - `middle_block.X.*` → `mid_block`
/// - `output_blocks.X.Y.*` → `up_blocks`
/// - `time_embed.*` → `time_embedding`
/// - `label_emb.*` → `add_embedding`
/// - `out.*` → `conv_norm_out`/`conv_out`
///
/// - Parameters:
///   - key: LDM-format weight key (after stripping `model.diffusion_model.` prefix)
///   - config: UNet configuration to determine block structure
/// - Returns: Equivalent diffusers-format key
func ldmUnetToDiffusersKey(_ key: String, config: UNetConfiguration) -> String {
    // time_embed → time_embedding
    if key.hasPrefix("time_embed.") {
        return key
            .replacingOccurrences(of: "time_embed.0.", with: "time_embedding.linear_1.")
            .replacingOccurrences(of: "time_embed.2.", with: "time_embedding.linear_2.")
    }

    // label_emb → add_embedding (SDXL text_time conditioning)
    if key.hasPrefix("label_emb.") {
        return key
            .replacingOccurrences(of: "label_emb.0.0.", with: "add_embedding.linear_1.")
            .replacingOccurrences(of: "label_emb.0.2.", with: "add_embedding.linear_2.")
    }

    // out.0 → conv_norm_out, out.2 → conv_out
    if key.hasPrefix("out.") {
        return key
            .replacingOccurrences(of: "out.0.", with: "conv_norm_out.")
            .replacingOccurrences(of: "out.2.", with: "conv_out.")
    }

    // input_blocks → conv_in / down_blocks
    if key.hasPrefix("input_blocks.") {
        return convertInputBlocks(key, config: config)
    }

    // middle_block → mid_block
    if key.hasPrefix("middle_block.") {
        return key
            .replacingOccurrences(of: "middle_block.0.", with: "mid_block.resnets.0.")
            .replacingOccurrences(of: "middle_block.1.", with: "mid_block.attentions.0.")
            .replacingOccurrences(of: "middle_block.2.", with: "mid_block.resnets.1.")
    }

    // output_blocks → up_blocks
    if key.hasPrefix("output_blocks.") {
        return convertOutputBlocks(key, config: config)
    }

    return key
}

/// Convert `input_blocks.X.Y.rest` to diffusers format.
private func convertInputBlocks(_ key: String, config: UNetConfiguration) -> String {
    // Parse input_blocks.X.Y.rest
    let withoutPrefix = String(key.dropFirst("input_blocks.".count))
    let parts = withoutPrefix.split(separator: ".", maxSplits: 2)
    guard parts.count >= 2,
          let blockIdx = Int(parts[0]),
          let layerIdx = Int(parts[1])
    else { return key }

    let rest = parts.count > 2 ? "." + parts[2] : ""

    // input_blocks.0.0 → conv_in
    if blockIdx == 0 && layerIdx == 0 {
        return "conv_in" + rest
    }

    // Build mapping from input block index to down_block structure
    let nBlocks = config.blockOutChannels.count
    var inputBlockCounter = 1  // start after conv_in at index 0

    for blockI in 0 ..< nBlocks {
        let hasAttn = config.downBlockTypes[blockI].contains("CrossAttn")
        let hasDownsample = blockI < nBlocks - 1
        let nLayers = config.layersPerBlock[blockI]

        for layerJ in 0 ..< nLayers {
            if blockIdx == inputBlockCounter {
                if layerIdx == 0 {
                    return "down_blocks.\(blockI).resnets.\(layerJ)" + rest
                }
                if hasAttn && layerIdx == 1 {
                    return "down_blocks.\(blockI).attentions.\(layerJ)" + rest
                }
            }
            inputBlockCounter += 1
        }

        if hasDownsample {
            if blockIdx == inputBlockCounter && layerIdx == 0 {
                return "down_blocks.\(blockI).downsamplers.0.conv" + rest
            }
            inputBlockCounter += 1
        }
    }

    return key  // fallback
}

/// Convert `output_blocks.X.Y.rest` to diffusers format.
private func convertOutputBlocks(_ key: String, config: UNetConfiguration) -> String {
    let withoutPrefix = String(key.dropFirst("output_blocks.".count))
    let parts = withoutPrefix.split(separator: ".", maxSplits: 2)
    guard parts.count >= 2,
          let blockIdx = Int(parts[0]),
          let layerIdx = Int(parts[1])
    else { return key }

    let rest = parts.count > 2 ? "." + parts[2] : ""

    // Output blocks go from deep to shallow, matching up_blocks order.
    // up_blocks are built with .reversed() enumeration in UNet.swift,
    // so up_blocks[0] is the deepest block.
    let nBlocks = config.blockOutChannels.count
    var outputBlockCounter = 0

    for blockI in 0 ..< nBlocks {
        // up_blocks[blockI] corresponds to reversed index
        let configIdx = nBlocks - 1 - blockI
        // Check from upBlockTypes (already reversed in config)
        let upHasAttn = config.upBlockTypes[configIdx].contains("CrossAttn")
        let hasUpsample = blockI < nBlocks - 1
        let nLayers = config.layersPerBlock[configIdx] + 1  // up blocks have +1 resnet

        for layerJ in 0 ..< nLayers {
            if blockIdx == outputBlockCounter {
                if layerIdx == 0 {
                    return "up_blocks.\(blockI).resnets.\(layerJ)" + rest
                }
                if upHasAttn && layerIdx == 1 {
                    return "up_blocks.\(blockI).attentions.\(layerJ)" + rest
                }
                // Upsample: last layer of the block
                if hasUpsample && layerJ == nLayers - 1 {
                    let upsampleLayerIdx = upHasAttn ? 2 : 1
                    if layerIdx == upsampleLayerIdx {
                        return "up_blocks.\(blockI).upsamplers.0.conv" + rest
                    }
                }
            }
            outputBlockCounter += 1
        }
    }

    return key  // fallback
}

/// Remap for single-file UNet: LDM format → diffusers format → MLX format.
///
/// Combines `ldmUnetToDiffusersKey` with `unetRemap`.
func ldmUnetRemap(key: String, value: MLXArray, config: UNetConfiguration) -> [(String, MLXArray)] {
    let diffusersKey = ldmUnetToDiffusersKey(key, config: config)
    return unetRemap(key: diffusersKey, value: value)
}

/// Remap for single-file OpenCLIP text encoder (text encoder 2 in SDXL).
///
/// OpenCLIP uses a different structure than standard CLIP:
/// - `transformer.resblocks.N.attn.in_proj_weight` → split into Q/K/V
/// - `transformer.resblocks.N.attn.out_proj.*` → attention output
/// - `transformer.resblocks.N.ln_1.*` → layer_norm1
/// - `transformer.resblocks.N.ln_2.*` → layer_norm2
/// - `transformer.resblocks.N.mlp.c_fc.*` → linear1
/// - `transformer.resblocks.N.mlp.c_proj.*` → linear2
func openClipRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key

    // Drop common prefixes
    if key.hasPrefix("text_model.") {
        key = String(key.dropFirst("text_model.".count))
    }

    // Map transformer.resblocks → layers
    key = key.replacingOccurrences(of: "transformer.resblocks.", with: "layers.")

    // Handle combined in_proj_weight/bias → split into Q, K, V
    if key.contains("attn.in_proj_weight") {
        let prefix = key.replacingOccurrences(of: "attn.in_proj_weight", with: "")
        let dim = value.dim(0) / 3
        let q = value[0 ..< dim]
        let k = value[dim ..< (2 * dim)]
        let v = value[(2 * dim)...]
        return [
            (prefix + "attention.query_proj.weight", q),
            (prefix + "attention.key_proj.weight", k),
            (prefix + "attention.value_proj.weight", v),
        ]
    }
    if key.contains("attn.in_proj_bias") {
        let prefix = key.replacingOccurrences(of: "attn.in_proj_bias", with: "")
        let dim = value.dim(0) / 3
        let q = value[0 ..< dim]
        let k = value[dim ..< (2 * dim)]
        let v = value[(2 * dim)...]
        return [
            (prefix + "attention.query_proj.bias", q),
            (prefix + "attention.key_proj.bias", k),
            (prefix + "attention.value_proj.bias", v),
        ]
    }

    // Map other attention keys
    key = key.replacingOccurrences(of: "attn.out_proj.", with: "attention.out_proj.")

    // Map layer norms
    key = key.replacingOccurrences(of: ".ln_1.", with: ".layer_norm1.")
    key = key.replacingOccurrences(of: ".ln_2.", with: ".layer_norm2.")

    // Map MLP
    key = key.replacingOccurrences(of: ".mlp.c_fc.", with: ".linear1.")
    key = key.replacingOccurrences(of: ".mlp.c_proj.", with: ".linear2.")

    // Map final layer norm
    key = key.replacingOccurrences(of: "ln_final.", with: "final_layer_norm.")

    // Map embeddings
    key = key.replacingOccurrences(of: "token_embedding.weight", with: "token_embedding.weight")
    key = key.replacingOccurrences(of: "positional_embedding", with: "position_embedding.weight")

    // Skip position_ids
    if key == "position_ids" { return [] }

    // text_projection → text_projection.weight
    if key == "text_projection" && value.ndim == 2 {
        return [("text_projection.weight", value)]
    }

    return [(key, value)]
}

// MARK: - Single-File Key Prefixes

/// Key prefixes for weights in a single-file SDXL checkpoint (Stability AI format).
///
/// A single .safetensors SDXL checkpoint contains all model components with these prefixes:
/// - `model.diffusion_model.*`  → UNet
/// - `conditioner.embedders.0.transformer.*`  → CLIP text encoder 1
/// - `conditioner.embedders.1.model.*`  → CLIP text encoder 2
/// - `first_stage_model.*`  → VAE (encoder + decoder)
private enum SingleFilePrefix {
    static let unet = "model.diffusion_model."
    static let textEncoder1 = "conditioner.embedders.0.transformer."
    static let textEncoder2 = "conditioner.embedders.1.model."
    static let vae = "first_stage_model."

    /// Alternative prefixes used by some checkpoint formats
    static let unetAlt = "unet."
    static let textEncoder1Alt = "text_encoder."
    static let textEncoder2Alt = "text_encoder_2."
}

/// Load Stable Diffusion XL from a single .safetensors checkpoint file.
///
/// This function handles the weight key remapping from the single-file format
/// (used by Stability AI, CivitAI models, etc.) to the MLX model structure.
///
/// **Tokenizer and scheduler config**: These are architecture-level files shared across
/// all SDXL models. This function loads them from the SDXL Turbo preset, which must be
/// downloaded first (`StableDiffusionConfiguration.presetSDXLTurbo.download()`).
///
/// - Parameters:
///   - url: Path to the .safetensors file containing the full model
///   - vaeUrl: Optional path to separate VAE .safetensors file
///   - hub: HubApi instance for tokenizer loading
///   - configuration: The stable diffusion configuration
///   - dType: Data type for weights
/// - Returns: A configured StableDiffusionXL model
public func loadStableDiffusionXLFromSingleFile(
    url checkpointUrl: URL,
    vaeUrl: URL? = nil,
    hub: HubApi = HubApi(),
    configuration: StableDiffusionConfiguration = .presetSDXLTurbo,
    dType: DType = LoadConfiguration().dType
) throws -> StableDiffusionXL {

    // ──────────────────────────────────────────────
    // 1. Load all weights from the checkpoint file
    // ──────────────────────────────────────────────
    print("Loading checkpoint from \(checkpointUrl.lastPathComponent)...")
    let allWeights = try loadArrays(url: checkpointUrl)

    // Detect the checkpoint format by looking at key prefixes
    let hasStabilityFormat = allWeights.keys.contains { $0.hasPrefix(SingleFilePrefix.unet) }
    let hasDiffusersFormat = allWeights.keys.contains { $0.hasPrefix(SingleFilePrefix.unetAlt) }

    let unetPrefix: String
    let te1Prefix: String
    let te2Prefix: String
    let vaePrefix: String

    if hasStabilityFormat {
        unetPrefix = SingleFilePrefix.unet
        te1Prefix = SingleFilePrefix.textEncoder1
        te2Prefix = SingleFilePrefix.textEncoder2
        vaePrefix = SingleFilePrefix.vae
    } else if hasDiffusersFormat {
        unetPrefix = SingleFilePrefix.unetAlt
        te1Prefix = SingleFilePrefix.textEncoder1Alt
        te2Prefix = SingleFilePrefix.textEncoder2Alt
        vaePrefix = SingleFilePrefix.vae  // same in both formats
    } else {
        // Fallback: try Stability format
        print("Warning: Could not detect checkpoint format. Trying Stability AI format...")
        unetPrefix = SingleFilePrefix.unet
        te1Prefix = SingleFilePrefix.textEncoder1
        te2Prefix = SingleFilePrefix.textEncoder2
        vaePrefix = SingleFilePrefix.vae
    }

    // ──────────────────────────────────────────────
    // 2. Load UNet configuration and weights
    // ──────────────────────────────────────────────
    // Use the SDXL Turbo config files as a reference (shared architecture)
    let sdxlTurbo = StableDiffusionConfiguration.presetSDXLTurbo

    print("Loading UNet configuration...")
    let unetConfig = try loadConfiguration(
        hub: hub, configuration: sdxlTurbo, key: .unetConfig, type: UNetConfiguration.self)
    let unet = UNetModel(configuration: unetConfig)

    // Extract and remap UNet weights
    print("Loading UNet weights...")
    let isLDM = hasStabilityFormat
    let unetWeights: [(String, MLXArray)] = allWeights
        .filter { $0.key.hasPrefix(unetPrefix) }
        .flatMap { key, value -> [(String, MLXArray)] in
            let stripped = String(key.dropFirst(unetPrefix.count))
            if isLDM {
                return ldmUnetRemap(key: stripped, value: value.asType(dType), config: unetConfig)
            } else {
                return unetRemap(key: stripped, value: value.asType(dType))
            }
        }
    print("  Remapped \(unetWeights.count) UNet weight tensors")
    try unet.update(parameters: ModuleParameters.unflattened(unetWeights), verify: .none)

    // ──────────────────────────────────────────────
    // 3. Load Text Encoder 1 (CLIP-L)
    // ──────────────────────────────────────────────
    print("Loading text encoder 1...")
    let te1Config = try loadConfiguration(
        hub: hub, configuration: sdxlTurbo, key: .textEncoderConfig,
        type: CLIPTextModelConfiguration.self)
    let textEncoder1 = CLIPTextModel(configuration: te1Config)

    let te1Weights: [(String, MLXArray)] = allWeights
        .filter { $0.key.hasPrefix(te1Prefix) }
        .flatMap { key, value -> [(String, MLXArray)] in
            let stripped = String(key.dropFirst(te1Prefix.count))
            return clipRemap(key: stripped, value: value.asType(dType))
        }
    try textEncoder1.update(parameters: ModuleParameters.unflattened(te1Weights), verify: .none)

    // ──────────────────────────────────────────────
    // 4. Load Text Encoder 2 (CLIP-G / OpenCLIP)
    // ──────────────────────────────────────────────
    print("Loading text encoder 2...")
    let te2Config = try loadConfiguration(
        hub: hub, configuration: sdxlTurbo, key: .textEncoderConfig2,
        type: CLIPTextModelConfiguration.self)
    let textEncoder2 = CLIPTextModel(configuration: te2Config)

    let te2Weights: [(String, MLXArray)] = allWeights
        .filter { $0.key.hasPrefix(te2Prefix) }
        .flatMap { key, value -> [(String, MLXArray)] in
            let stripped = String(key.dropFirst(te2Prefix.count))
            if isLDM {
                // OpenCLIP format: combined in_proj, resblocks, etc.
                return openClipRemap(key: stripped, value: value.asType(dType))
            } else {
                return clipRemap(key: stripped, value: value.asType(dType))
            }
        }
    print("  Remapped \(te2Weights.count) text encoder 2 weight tensors")
    try textEncoder2.update(parameters: ModuleParameters.unflattened(te2Weights), verify: .none)

    // ──────────────────────────────────────────────
    // 5. Load VAE
    // ──────────────────────────────────────────────
    print("Loading VAE...")
    let autoencoderConfig = AutoencoderConfiguration()
    let autoencoder = Autoencoder(configuration: autoencoderConfig)

    if let vaeUrl {
        // Load from separate VAE file (e.g., sdxl.vae.safetensors)
        print("Using separate VAE file: \(vaeUrl.lastPathComponent)")
        try loadWeights(url: vaeUrl, model: autoencoder, mapper: vaeRemap, dType: .float32)
    } else {
        // Extract VAE weights from the checkpoint
        let vaeWeights: [(String, MLXArray)] = allWeights
            .filter { $0.key.hasPrefix(vaePrefix) }
            .flatMap { key, value -> [(String, MLXArray)] in
                let stripped = String(key.dropFirst(vaePrefix.count))
                return vaeRemap(key: stripped, value: value.asType(.float32))
            }
        try autoencoder.update(
            parameters: ModuleParameters.unflattened(vaeWeights), verify: .none)
    }

    // ──────────────────────────────────────────────
    // 6. Load diffusion config and sampler
    // ──────────────────────────────────────────────
    print("Loading scheduler configuration...")
    let diffusionConfig = try loadDiffusionConfiguration(hub: hub, configuration: sdxlTurbo)
    let sampler = SimpleEulerAncestralSampler(configuration: diffusionConfig)

    // ──────────────────────────────────────────────
    // 7. Load tokenizers (shared across all SDXL models)
    // ──────────────────────────────────────────────
    print("Loading tokenizers...")
    let tokenizer1 = try loadTokenizer(hub: hub, configuration: sdxlTurbo)
    let tokenizer2 = try loadTokenizer(
        hub: hub, configuration: sdxlTurbo,
        vocabulary: .tokenizerVocabulary2, merges: .tokenizerMerges2)

    // ──────────────────────────────────────────────
    // 8. Assemble the model
    // ──────────────────────────────────────────────
    print("Assembling StableDiffusionXL model...")
    let model = try StableDiffusionXL(
        hub: hub, configuration: configuration, dType: dType,
        diffusionConfiguration: diffusionConfig,
        unet: unet,
        textEncoder: textEncoder1,
        autoencoder: autoencoder,
        sampler: sampler,
        tokenizer: tokenizer1,
        textEncoder2: textEncoder2,
        tokenizer2: tokenizer2
    )

    print("Model loaded successfully!")
    return model
}

/// Load and fuse LoRA weights from a .safetensors file into a StableDiffusion model.
///
/// This is a convenience wrapper that loads LoRA weights and fuses them into the model.
/// See ``Lora.swift`` for the implementation details.
///
/// - Parameters:
///   - sd: The StableDiffusion model to fuse LoRA into
///   - loraUrl: Path to the .safetensors file containing LoRA weights
///   - scale: Scale factor for LoRA weights (default 1.0)
// loadAndFuseLora is defined in Lora.swift

// MARK: - Tokenizer

func loadTokenizer(
    hub: HubApi, configuration: StableDiffusionConfiguration,
    vocabulary: FileKey = .tokenizerVocabulary, merges: FileKey = .tokenizerMerges
) throws -> CLIPTokenizer {
    let vocabularyURL = resolve(hub: hub, configuration: configuration, key: vocabulary)
    let mergesURL = resolve(hub: hub, configuration: configuration, key: merges)

    let vocabulary = try JSONDecoder().decode(
        [String: Int].self, from: Data(contentsOf: vocabularyURL))
    let merges = try String(contentsOf: mergesURL)
        .components(separatedBy: .newlines)
        // first line is a comment
        .dropFirst()
        .filter { !$0.isEmpty }

    return CLIPTokenizer(merges: merges, vocabulary: vocabulary)
}
