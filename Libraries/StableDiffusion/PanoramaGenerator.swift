// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Panorama Parameters

/// Parameters for generating a 360° panorama image.
///
/// These parameters control the generation of seamless panoramic images with X-axis tiling.
/// Default values match the reference Python script (DreamShaperXL + 360Redmond LoRA).
public struct PanoramaParameters: Sendable {
    /// The text prompt describing the desired image
    public var prompt: String

    /// Negative prompt to avoid certain elements
    public var negativePrompt: String

    /// Output width in pixels (must be multiple of 64)
    public var width: Int

    /// Output height in pixels (must be multiple of 64)
    public var height: Int

    /// Number of inference steps
    public var steps: Int

    /// Classifier-free guidance scale
    public var cfgScale: Float

    /// Random seed for reproducibility
    public var seed: UInt64?

    /// Denoising strength (1.0 = full denoising)
    public var denoise: Float

    /// Enable seamless tiling on X-axis (horizontal wrapping for 360°)
    public var tileX: Bool

    /// Enable seamless tiling on Y-axis (vertical wrapping)
    public var tileY: Bool

    /// Create panorama parameters
    ///
    /// Default values match the reference Python panorama generator:
    /// - 2048×1024 output (equirectangular 2:1 aspect ratio)
    /// - 8 inference steps with cfg=3.0
    /// - Seamless X-axis tiling for 360° wrap
    ///
    /// - Parameters:
    ///   - prompt: Text description of the image
    ///   - negativePrompt: Elements to avoid (default includes common artifacts)
    ///   - width: Output width (default 2048, rounded to multiple of 64)
    ///   - height: Output height (default 1024, rounded to multiple of 64)
    ///   - steps: Number of inference steps (default 8)
    ///   - cfgScale: Guidance scale (default 3.0)
    ///   - seed: Random seed for reproducibility
    ///   - tileX: Enable X-axis seamless tiling (default true)
    ///   - tileY: Enable Y-axis seamless tiling (default false)
    public init(
        prompt: String = "",
        negativePrompt: String = "boring, text, signature, watermark, low quality, bad quality, grainy, blurry, long neck, closed eyes",
        width: Int = 2048,
        height: Int = 1024,
        steps: Int = 8,
        cfgScale: Float = 3.0,
        seed: UInt64? = nil,
        tileX: Bool = true,
        tileY: Bool = false
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        // Ensure dimensions are multiples of 64
        self.width = width - (width % 64)
        self.height = height - (height % 64)
        self.steps = steps
        self.cfgScale = cfgScale
        self.seed = seed ?? UInt64(Date.timeIntervalSinceReferenceDate * 1000)
        self.denoise = 1.0
        self.tileX = tileX
        self.tileY = tileY
    }
}

// MARK: - Panorama Generator

/// A specialized generator for creating 360° seamless panoramic images.
///
/// This generator enables seamless tiling on the X-axis, allowing the left and right edges
/// of the generated image to seamlessly connect into a circular panorama.
///
/// The implementation applies circular padding to the latent tensor at the UNet boundary,
/// which encourages the model to generate content that wraps seamlessly.
///
/// ## Usage with Local Models
///
/// ```swift
/// // 1. Download SDXL Turbo first (for tokenizer/scheduler files)
/// let sdxlTurbo = StableDiffusionConfiguration.presetSDXLTurbo
/// try await sdxlTurbo.download()
///
/// // 2. Load custom checkpoint with LoRA
/// let paths = LocalModelPaths(
///     checkpoint: URL(filePath: "/path/to/dreamshaperXL.safetensors"),
///     vae: URL(filePath: "/path/to/sdxl.vae.safetensors"),
///     loras: [
///         (url: URL(filePath: "/path/to/360Redmond.safetensors"), scale: 1.0)
///     ]
/// )
///
/// let sd = try loadStableDiffusionXLFromSingleFile(
///     url: paths.checkpoint, vaeUrl: paths.vae, dType: .float16
/// )
///
/// // 3. Fuse LoRA weights
/// for (loraUrl, scale) in paths.loras {
///     try loadAndFuseLora(sd, loraUrl: loraUrl, scale: scale)
/// }
///
/// // 4. Create panorama generator and generate
/// let generator = PanoramaGenerator(sd)
/// let parameters = PanoramaParameters(
///     prompt: "Glowing mushrooms around pyramids, equirectangular, 360 panorama",
///     steps: 8,
///     cfgScale: 3.0
/// )
///
/// let latents = generator.generateLatents(parameters: parameters)
/// var lastXt: MLXArray?
/// for xt in latents {
///     eval(xt)
///     lastXt = xt
/// }
///
/// if let lastXt {
///     let decoded = generator.decode(xt: lastXt[0])
///     let raster = (decoded * 255).asType(.uint8).squeezed()
///     try Image(raster).save(url: URL(filePath: "/tmp/panorama_360.png"))
/// }
/// ```
public class PanoramaGenerator {

    let sd: StableDiffusionXL

    /// Width of the output image in pixels
    public var width: Int { latentSize[1] * 8 }

    /// Height of the output image in pixels
    public var height: Int { latentSize[0] * 8 }

    /// Latent size (image is 8x larger)
    private let latentSize: [Int]

    /// Whether seamless X-axis tiling is enabled
    private var tileX: Bool = true

    /// Whether seamless Y-axis tiling is enabled
    private var tileY: Bool = false

    /// Create a panorama generator from an existing StableDiffusionXL model
    ///
    /// - Parameters:
    ///   - sd: The underlying SDXL model
    ///   - width: Output image width (default 2048)
    ///   - height: Output image height (default 1024)
    public init(_ sd: StableDiffusionXL, width: Int = 2048, height: Int = 1024) {
        self.sd = sd
        // Latent size is 1/8 of image size (due to VAE encoding)
        self.latentSize = [height / 8, width / 8]
    }

    /// Generate latents for a 360° panorama with seamless X-axis tiling.
    ///
    /// This method replaces all Conv2d layers in the UNet and VAE decoder with
    /// ``SeamlessConv2d`` instances that use circular padding on the configured axes.
    /// This matches the Python reference implementation where every Conv2d's
    /// `_conv_forward` is patched for seamless tiling.
    ///
    /// - Parameter parameters: Panorama generation parameters
    /// - Returns: An iterator that produces latent images
    public func generateLatents(parameters: PanoramaParameters) -> DenoiseIterator {
        // Store tiling configuration
        self.tileX = parameters.tileX
        self.tileY = parameters.tileY

        // Create standard evaluation parameters from panorama parameters
        let evaluateParams = EvaluateParameters(
            cfgWeight: parameters.cfgScale,
            steps: parameters.steps,
            imageCount: 1,
            decodingBatchSize: 1,
            latentSize: [parameters.height / 8, parameters.width / 8],
            seed: parameters.seed,
            prompt: parameters.prompt,
            negativePrompt: parameters.negativePrompt
        )

        // Replace all Conv2d layers with SeamlessConv2d in UNet and VAE decoder.
        // This patches every convolution to use circular padding on the configured
        // axes, matching the Python enable_seamless_tiling() behavior.
        enableSeamlessTiling(sd.unet, tileX: parameters.tileX, tileY: parameters.tileY)
        enableSeamlessTiling(sd.autoencoder.decoder, tileX: parameters.tileX, tileY: parameters.tileY)

        return sd.generateLatents(parameters: evaluateParams)
    }

    /// Generate latents for a 360° panorama with seamless tiling on both axes.
    ///
    /// This creates tiles that can be seamlessly repeated in both directions.
    ///
    /// - Parameter parameters: Panorama generation parameters
    /// - Returns: An iterator that produces latent images
    func generateTiledLatents(parameters: PanoramaParameters) -> DenoiseIterator {
        var params = parameters
        params.tileX = true
        params.tileY = true
        return generateLatents(parameters: params)
    }
}

// MARK: - TextToImageGenerator Conformance

extension PanoramaGenerator: TextToImageGenerator {

    public func ensureLoaded() {
        sd.ensureLoaded()
    }

    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.sd.autoencoder
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }

    public func decode(xt: MLXArray) -> MLXArray {
        sd.decode(xt: xt)
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        // Convert to panorama parameters
        let panoramaParams = PanoramaParameters(
            prompt: parameters.prompt,
            width: parameters.latentSize[1] * 8,
            height: parameters.latentSize[0] * 8,
            steps: parameters.steps,
            cfgScale: parameters.cfgWeight,
            seed: parameters.seed
        )
        return generateLatents(parameters: panoramaParams)
    }
}

// MARK: - Convenience Functions

/// Enable seamless X-axis tiling on a model and its components.
///
/// This function registers tiling configuration for a model. The actual circular
/// padding is applied at the inference level when the model is called.
///
/// - Parameters:
///   - sd: The StableDiffusion model
func enablePanoramaTiling(_ sd: StableDiffusion) {
    enableSeamlessTiling(sd.unet, tileX: true, tileY: false)
    enableSeamlessTiling(sd.autoencoder.decoder, tileX: true, tileY: false)
}

/// Enable full seamless tiling (both axes) on a model.
///
/// - Parameter sd: The StableDiffusion model
func enableFullTiling(_ sd: StableDiffusion) {
    enableSeamlessTiling(sd.unet, tileX: true, tileY: true)
    enableSeamlessTiling(sd.autoencoder.decoder, tileX: true, tileY: true)
}

// MARK: - Local Model Convenience

extension StableDiffusionConfiguration {

    /// Create a panorama-optimized configuration for loading from local files.
    ///
    /// This sets up the configuration with panorama-appropriate defaults:
    /// - 8 inference steps
    /// - CFG scale of 3.0
    /// - 2048×1024 latent size (256×128 latent)
    ///
    /// - Parameters:
    ///   - paths: Local model file paths
    /// - Returns: A configuration for panorama generation
    public static func panoramaModel(
        _ paths: LocalModelPaths
    ) -> StableDiffusionConfiguration {
        return localModel(paths) {
            EvaluateParameters(
                cfgWeight: 3.0,
                steps: 8,
                imageCount: 1,
                decodingBatchSize: 1,
                latentSize: [128, 256],  // 1024×2048 output
                prompt: "",
                negativePrompt:
                    "boring, text, signature, watermark, low quality, bad quality, grainy, blurry, long neck, closed eyes"
            )
        }
    }
}
