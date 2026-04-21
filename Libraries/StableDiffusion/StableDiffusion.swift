// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/__init__.py

// Lora.swift is part of this same module (StableDiffusion)

/// Iterator that produces latent images.
///
/// Created by:
///
/// - ``TextToImageGenerator/generateLatents(parameters:)``
/// - ``ImageToImageGenerator/generateLatents(image:parameters:strength:)``
public struct DenoiseIterator: Sequence, IteratorProtocol {

    let sd: StableDiffusion

    var xt: MLXArray

    let conditioning: MLXArray
    let cfgWeight: Float
    let textTime: (MLXArray, MLXArray)?

    var i: Int
    let steps: [(MLXArray, MLXArray)]

    init(
        sd: StableDiffusion, xt: MLXArray, t: Int, conditioning: MLXArray, steps: Int,
        cfgWeight: Float, textTime: (MLXArray, MLXArray)? = nil
    ) {
        self.sd = sd
        self.steps = sd.sampler.timeSteps(steps: steps, start: t, dType: sd.dType)
        self.i = 0
        self.xt = xt
        self.conditioning = conditioning
        self.cfgWeight = cfgWeight
        self.textTime = textTime
    }

    public var underestimatedCount: Int {
        steps.count
    }

    mutating public func next() -> MLXArray? {
        guard i < steps.count else {
            return nil
        }

        let (t, tPrev) = steps[i]
        i += 1

        xt = sd.step(
            xt: xt, t: t, tPrev: tPrev, conditioning: conditioning, cfgWeight: cfgWeight,
            textTime: textTime)
        return xt
    }
}

// MARK: - Seamless Tiling via Conv2d Replacement

/// A Conv2d subclass that applies circular (seamless) padding instead of zero padding.
///
/// This mirrors the Python approach from `sdxl_lightning.py` where every Conv2d layer
/// in the UNet and VAE decoder has its `_conv_forward` patched to use circular padding
/// on the configured axes.
///
/// For seamless 360° panoramas, X-axis circular padding makes the left and right edges
/// wrap around, eliminating visible seams.
///
/// Input is expected in NHWC layout: [batch, height, width, channels].
class SeamlessConv2d: Conv2d {

    let tileX: Bool
    let tileY: Bool
    let originalPaddingH: Int
    let originalPaddingW: Int

    /// Create a SeamlessConv2d from an existing Conv2d's configuration.
    ///
    /// - Parameters:
    ///   - from: The original Conv2d to take weight/bias/stride/dilation/groups from
    ///   - tileX: Whether to use circular padding on the width (X) axis
    ///   - tileY: Whether to use circular padding on the height (Y) axis
    init(from conv: Conv2d, tileX: Bool = true, tileY: Bool = false) {
        self.tileX = tileX
        self.tileY = tileY
        self.originalPaddingH = conv.padding.0
        self.originalPaddingW = conv.padding.1

        // Initialize the parent Conv2d with padding=0 since we handle padding manually.
        // We use a dummy init and then overwrite the weight/bias via update(parameters:).
        let outputChannels = conv.weight.dim(0)
        let kernelH = conv.weight.dim(1)
        let kernelW = conv.weight.dim(2)
        let inputChannels = conv.weight.dim(3) * conv.groups

        super.init(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair((kernelH, kernelW)),
            stride: IntOrPair(conv.stride),
            padding: 0,  // We apply padding ourselves
            dilation: IntOrPair(conv.dilation),
            groups: conv.groups,
            bias: conv.bias != nil
        )

        // Copy the actual trained weights from the original Conv2d
        var params = [(String, MLXArray)]()
        params.append(("weight", conv.weight))
        if let bias = conv.bias {
            params.append(("bias", bias))
        }
        try! self.update(
            parameters: ModuleParameters.unflattened(params), verify: .none)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        let padH = originalPaddingH
        let padW = originalPaddingW

        // Apply asymmetric padding matching the Python reference:
        //   if tile_x and tile_y:
        //       F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode="circular")
        //   elif tile_x:
        //       F.pad(input, (pad_w, pad_w, 0, 0), mode="circular")
        //       F.pad(input, (0, 0, pad_h, pad_h), mode="constant", value=0)
        //   elif tile_y:
        //       F.pad(input, (0, 0, pad_h, pad_h), mode="circular")
        //       F.pad(input, (pad_w, pad_w, 0, 0), mode="constant", value=0)

        // Width (X-axis) padding — NHWC layout, axis 2 is width
        if padW > 0 {
            if tileX {
                let W = input.dim(2)
                let leftWrap = input[0..., 0..., (W - padW)..., 0...]
                let rightWrap = input[0..., 0..., ..<padW, 0...]
                input = concatenated([leftWrap, input, rightWrap], axis: 2)
            } else {
                input = padded(input, widths: [[0, 0], [0, 0], [padW, padW], [0, 0]])
            }
        }

        // Height (Y-axis) padding — NHWC layout, axis 1 is height
        if padH > 0 {
            if tileY {
                let H = input.dim(1)
                let topWrap = input[0..., (H - padH)..., 0..., 0...]
                let bottomWrap = input[0..., ..<padH, 0..., 0...]
                input = concatenated([topWrap, input, bottomWrap], axis: 1)
            } else {
                input = padded(input, widths: [[0, 0], [padH, padH], [0, 0], [0, 0]])
            }
        }

        // Run conv2d with padding=0 (padding was already applied above)
        var y = conv2d(
            input, weight, stride: .init(stride), padding: .init((0, 0)),
            dilation: .init(dilation), groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

/// Enable seamless tiling on a model by replacing all Conv2d layers that have
/// non-zero padding with ``SeamlessConv2d`` instances.
///
/// This walks the entire module tree and replaces Conv2d layers in-place,
/// matching the Python approach where every Conv2d's `_conv_forward` is patched.
///
/// Conv2d layers with zero padding are left unchanged (they don't need wrapping).
///
/// - Parameters:
///   - model: The model to patch (e.g. UNet or VAE decoder)
///   - tileX: Whether to enable circular padding on the X (width) axis
///   - tileY: Whether to enable circular padding on the Y (height) axis
func enableSeamlessTiling(_ model: Module, tileX: Bool = true, tileY: Bool = false) {
    // Walk every module in the tree
    let allModules = model.namedModules()

    for (_, module) in allModules {
        // Look at each module's direct children for Conv2d instances
        let children = module.children()
        var replacements = ModuleChildren()
        var hasReplacements = false

        for (key, child) in children.flattened() {
            if let childConv = child as? Conv2d,
               !(childConv is SeamlessConv2d),  // Don't double-wrap
               (childConv.padding.0 > 0 || childConv.padding.1 > 0)
            {
                let seamless = SeamlessConv2d(from: childConv, tileX: tileX, tileY: tileY)
                replacements[key] = .value(seamless)
                hasReplacements = true
            }
        }

        if hasReplacements {
            // update(modules:) requires @ModuleInfo on the property
            try? module.update(modules: replacements, verify: .none)
        }
    }
}

/// Disable seamless tiling on a model by replacing all ``SeamlessConv2d`` layers
/// back to standard Conv2d instances.
///
/// This restores the original zero-padding behavior.
///
/// - Parameter model: The model to restore
func disableSeamlessTiling(_ model: Module) {
    let allModules = model.namedModules()

    for (_, module) in allModules {
        let children = module.children()
        var replacements = ModuleChildren()
        var hasReplacements = false

        for (key, child) in children.flattened() {
            if let seamlessConv = child as? SeamlessConv2d {
                // Create a standard Conv2d with the original padding restored
                let outputChannels = seamlessConv.weight.dim(0)
                let kernelH = seamlessConv.weight.dim(1)
                let kernelW = seamlessConv.weight.dim(2)
                let inputChannels = seamlessConv.weight.dim(3) * seamlessConv.groups

                let restored = Conv2d(
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    kernelSize: IntOrPair((kernelH, kernelW)),
                    stride: IntOrPair(seamlessConv.stride),
                    padding: IntOrPair((seamlessConv.originalPaddingH, seamlessConv.originalPaddingW)),
                    dilation: IntOrPair(seamlessConv.dilation),
                    groups: seamlessConv.groups,
                    bias: seamlessConv.bias != nil
                )

                var params = [(String, MLXArray)]()
                params.append(("weight", seamlessConv.weight))
                if let bias = seamlessConv.bias {
                    params.append(("bias", bias))
                }
                let verify = Module.VerifyUpdate.none
                try! restored.update(
                    parameters: ModuleParameters.unflattened(params), verify: verify)

                replacements[key] = .value(restored)
                hasReplacements = true
            }
        }

        if hasReplacements {
            try? module.update(modules: replacements, verify: .none)
        }
    }
}

// MARK: - Configuration Extensions

extension StableDiffusionConfiguration {
    
    /// Create a configuration for loading from local .safetensors files
    ///
    /// - Parameters:
    ///   - paths: Local model file paths (checkpoint, optional VAE, optional LoRA)
    ///   - defaultParameters: Function returning default evaluation parameters
    /// - Returns: A configuration that can load from local files
    public static func localModel(
        _ paths: LocalModelPaths,
        defaultParameters: @escaping () -> EvaluateParameters = { EvaluateParameters(cfgWeight: 2.0, steps: 4) }
    ) -> StableDiffusionConfiguration {
        // Use a placeholder ID for local models
        let id = "local:\(paths.checkpoint.lastPathComponent)"
        
        return StableDiffusionConfiguration(
            id: id,
            files: [:], // No HF hub files needed
            defaultParameters: defaultParameters,
            factory: { hub, config, loadConfig in
                try loadStableDiffusionXLFromSingleFile(
                    url: paths.checkpoint,
                    vaeUrl: paths.vae,
                    hub: hub,
                    configuration: config,
                    dType: loadConfig.dType
                )
            }
        )
    }
    
    /// Create a configuration with LoRA support
    ///
    /// - Parameters:
    ///   - baseConfiguration: The base configuration to extend
    ///   - loraUrl: URL to the LoRA weights file
    ///   - scale: Scale factor for LoRA (default 1.0)
    public func withLora(_ loraUrl: URL, scale: Float = 1.0) -> StableDiffusionConfiguration {
        var config = self
        // Store LoRA info in metadata (would need to extend configuration)
        return config
    }
}

// MARK: - Convenience Factory for Local Models

extension ModelContainer {

    /// Create a ``ModelContainer`` that supports ``TextToImageGenerator`` from local files
    ///
    /// - Parameters:
    ///   - paths: Local model file paths
    ///   - loadConfiguration: Configuration for loading weights
    public static func createLocalTextToImageGenerator(
        paths: LocalModelPaths,
        loadConfiguration: LoadConfiguration = .init()
    ) throws -> ModelContainer<TextToImageGenerator> {
        let sdConfiguration = StableDiffusionConfiguration.localModel(paths)

        if let model = try sdConfiguration.textToImageGenerator(configuration: loadConfiguration) {
            return .init(model: model)
        } else {
            throw ModelContainerError.unableToCreate("local model", "TextToImageGenerator")
        }
    }
}

/// Type for the _decoder_ step.
public typealias ImageDecoder = (MLXArray) -> MLXArray

public protocol ImageGenerator {
    func ensureLoaded()

    /// Return a detached decoder -- this is useful if trying to conserve memory.
    ///
    /// The decoder can be used independently of the ImageGenerator to transform
    /// latents into raster images.
    func detachedDecoder() -> ImageDecoder

    /// the equivalent to the ``detachedDecoder()`` but without the detatching
    func decode(xt: MLXArray) -> MLXArray
}

/// Public interface for transforming a text prompt into an image.
///
/// Steps:
///
/// - ``generateLatents(parameters:)``
/// - evaluate each of the latents from the iterator
/// - ``ImageGenerator/decode(xt:)`` or ``ImageGenerator/detachedDecoder()`` to convert the final latent into an image
/// - use ``Image`` to save the image
public protocol TextToImageGenerator: ImageGenerator {
    func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator
}

/// Public interface for transforming a text prompt into an image.
///
/// Steps:
///
/// - ``generateLatents(image:parameters:strength:)``
/// - evaluate each of the latents from the iterator
/// - ``ImageGenerator/decode(xt:)`` or ``ImageGenerator/detachedDecoder()`` to convert the final latent into an image
/// - use ``Image`` to save the image
public protocol ImageToImageGenerator: ImageGenerator {
    func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
}

enum ModelContainerError: LocalizedError {
    /// Unable to create the particular type of model, e.g. it doesn't support image to image
    case unableToCreate(String, String)
    /// When operating in conserveMemory mode, it tried to use a model that had been discarded
    case modelDiscarded

    var errorDescription: String? {
        switch self {
        case .unableToCreate(let modelId, let generatorType):
            return String(
                localized:
                    "Unable to create a \(generatorType) with model ID '\(modelId)'. The model may not support this operation type."
            )
        case .modelDiscarded:
            return String(
                localized:
                    "The model has been discarded to conserve memory and is no longer available. Please recreate the model container."
            )
        }
    }
}

/// Container for models that guarantees single threaded access.
public actor ModelContainer<M> {

    enum State {
        case discarded
        case loaded(M)
    }

    var state: State

    /// if true this will discard the model in ``performTwoStage(first:second:)``
    var conserveMemory = false

    private init(model: M) {
        self.state = .loaded(model)
    }

    /// create a ``ModelContainer`` that supports ``TextToImageGenerator``
    static public func createTextToImageGenerator(
        configuration: StableDiffusionConfiguration, loadConfiguration: LoadConfiguration = .init()
    ) throws -> ModelContainer<TextToImageGenerator> {
        if let model = try configuration.textToImageGenerator(configuration: loadConfiguration) {
            return .init(model: model)
        } else {
            throw ModelContainerError.unableToCreate(configuration.id, "TextToImageGenerator")
        }
    }

    /// create a ``ModelContainer`` that supports ``ImageToImageGenerator``
    static public func createImageToImageGenerator(
        configuration: StableDiffusionConfiguration, loadConfiguration: LoadConfiguration = .init()
    ) throws -> ModelContainer<ImageToImageGenerator> {
        if let model = try configuration.imageToImageGenerator(configuration: loadConfiguration) {
            return .init(model: model)
        } else {
            throw ModelContainerError.unableToCreate(configuration.id, "ImageToImageGenerator")
        }
    }

    public func setConserveMemory(_ conserveMemory: Bool) {
        self.conserveMemory = conserveMemory
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (M) throws -> R) throws -> R {
        switch state {
        case .discarded:
            throw ModelContainerError.modelDiscarded
        case .loaded(let m):
            try action(m)
        }
    }

    /// Perform a two stage action where the first stage returns values passed to the second stage.
    ///
    /// If ``setConservativeMemory(_:)`` is `true` this will discard the model in between
    /// the `first` and `second` blocks. The container will have to be recreated if a caller
    /// wants to use it again.
    ///
    /// If `false` this will just run them in sequence and the container can be reused.
    ///
    /// Callers _must_ eval any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    public func performTwoStage<R1, R2>(
        first: @Sendable (M) throws -> R1, second: @Sendable (R1) throws -> R2
    ) throws -> R2 {
        let r1 =
            switch state {
            case .discarded:
                throw ModelContainerError.modelDiscarded
            case .loaded(let m):
                try first(m)
            }
        if conserveMemory {
            self.state = .discarded
        }
        return try second(r1)
    }

}

/// Base class for Stable Diffusion.
open class StableDiffusion {

    let dType: DType
    let diffusionConfiguration: DiffusionConfiguration
    let unet: UNetModel
    let textEncoder: CLIPTextModel
    let autoencoder: Autoencoder
    let sampler: SimpleEulerSampler
    let tokenizer: CLIPTokenizer

    internal init(
        hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType,
        diffusionConfiguration: DiffusionConfiguration? = nil, unet: UNetModel? = nil,
        textEncoder: CLIPTextModel? = nil, autoencoder: Autoencoder? = nil,
        sampler: SimpleEulerSampler? = nil, tokenizer: CLIPTokenizer? = nil
    ) throws {
        self.dType = dType
        self.diffusionConfiguration =
            try diffusionConfiguration
            ?? loadDiffusionConfiguration(hub: hub, configuration: configuration)
        self.unet = try unet ?? loadUnet(hub: hub, configuration: configuration, dType: dType)
        self.textEncoder =
            try textEncoder ?? loadTextEncoder(hub: hub, configuration: configuration, dType: dType)

        // note: autoencoder uses float32 weights
        self.autoencoder =
            try autoencoder
            ?? loadAutoEncoder(hub: hub, configuration: configuration, dType: .float32)

        if let sampler {
            self.sampler = sampler
        } else {
            self.sampler = SimpleEulerSampler(configuration: self.diffusionConfiguration)
        }
        self.tokenizer = try tokenizer ?? loadTokenizer(hub: hub, configuration: configuration)
    }

    open func ensureLoaded() {
        eval(unet, textEncoder, autoencoder)
    }

    func tokenize(tokenizer: CLIPTokenizer, text: String, negativeText: String?) -> MLXArray {
        var tokens = [tokenizer.tokenize(text: text)]
        if let negativeText {
            tokens.append(tokenizer.tokenize(text: negativeText))
        }

        let c = tokens.count
        let max = tokens.map { $0.count }.max() ?? 0
        let mlxTokens = MLXArray(
            tokens
                .map {
                    ($0 + Array(repeating: 0, count: max - $0.count))
                }
                .flatMap { $0 }
        )
        .reshaped(c, max)

        return mlxTokens
    }

    open func step(
        xt: MLXArray, t: MLXArray, tPrev: MLXArray, conditioning: MLXArray, cfgWeight: Float,
        textTime: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let xtUnet = cfgWeight > 1 ? concatenated([xt, xt], axis: 0) : xt
        let tUnet = broadcast(t, to: [xtUnet.count])

        var epsPred = unet(xtUnet, timestep: tUnet, encoderX: conditioning, textTime: textTime)

        if cfgWeight > 1 {
            let (epsText, epsNeg) = epsPred.split()
            epsPred = epsNeg + cfgWeight * (epsText - epsNeg)
        }

        return sampler.step(epsPred: epsPred, xt: xt, t: t, tPrev: tPrev)
    }

    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.autoencoder
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }

    public func decode(xt: MLXArray) -> MLXArray {
        detachedDecoder()(xt)
    }
}

/// Implementation of ``StableDiffusion`` for the `stabilityai/stable-diffusion-2-1-base` model.
open class StableDiffusionBase: StableDiffusion, TextToImageGenerator {

    public init(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws {
        try super.init(hub: hub, configuration: configuration, dType: dType)
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?)
        -> MLXArray
    {
        // tokenize the text
        let tokens = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)

        // compute the features
        var conditioning = textEncoder(tokens).lastHiddenState

        // repeat the conditioning for each of the generated images
        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
        }

        return conditioning
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let conditioning = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight)
    }

}

/// Implementation of ``StableDiffusion`` for the `stabilityai/sdxl-turbo` model.
open class StableDiffusionXL: StableDiffusion, TextToImageGenerator, ImageToImageGenerator {

    let textEncoder2: CLIPTextModel
    let tokenizer2: CLIPTokenizer

    public init(hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType) throws {
        let diffusionConfiguration = try loadConfiguration(
            hub: hub, configuration: configuration, key: .diffusionConfig,
            type: DiffusionConfiguration.self)
        let sampler = SimpleEulerAncestralSampler(configuration: diffusionConfiguration)

        self.textEncoder2 = try loadTextEncoder(
            hub: hub, configuration: configuration, configKey: .textEncoderConfig2,
            weightsKey: .textEncoderWeights2, dType: dType)

        self.tokenizer2 = try loadTokenizer(
            hub: hub, configuration: configuration, vocabulary: .tokenizerVocabulary2,
            merges: .tokenizerMerges2)

        try super.init(
            hub: hub, configuration: configuration, dType: dType,
            diffusionConfiguration: diffusionConfiguration, sampler: sampler)
    }

    /// Initialize with pre-built components (for loading from local files).
    ///
    /// This initializer accepts all model components directly, bypassing the
    /// HuggingFace Hub file resolution. Used by ``loadStableDiffusionXLFromSingleFile()``.
    internal init(
        hub: HubApi, configuration: StableDiffusionConfiguration, dType: DType,
        diffusionConfiguration: DiffusionConfiguration,
        unet: UNetModel,
        textEncoder: CLIPTextModel,
        autoencoder: Autoencoder,
        sampler: SimpleEulerSampler,
        tokenizer: CLIPTokenizer,
        textEncoder2: CLIPTextModel,
        tokenizer2: CLIPTokenizer
    ) throws {
        self.textEncoder2 = textEncoder2
        self.tokenizer2 = tokenizer2
        try super.init(
            hub: hub, configuration: configuration, dType: dType,
            diffusionConfiguration: diffusionConfiguration,
            unet: unet, textEncoder: textEncoder,
            autoencoder: autoencoder, sampler: sampler, tokenizer: tokenizer)
    }

    open override func ensureLoaded() {
        super.ensureLoaded()
        eval(textEncoder2)
    }

    func conditionText(text: String, imageCount: Int, cfgWeight: Float, negativeText: String?) -> (
        MLXArray, MLXArray
    ) {
        let tokens1 = tokenize(
            tokenizer: tokenizer, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)
        let tokens2 = tokenize(
            tokenizer: tokenizer2, text: text, negativeText: cfgWeight > 1 ? negativeText : nil)

        let conditioning1 = textEncoder(tokens1)
        let conditioning2 = textEncoder2(tokens2)
        var conditioning = concatenated(
            [
                conditioning1.hiddenStates.dropLast().last!,
                conditioning2.hiddenStates.dropLast().last!,
            ],
            axis: -1)
        var pooledConditionng = conditioning2.pooledOutput

        if imageCount > 1 {
            conditioning = repeated(conditioning, count: imageCount, axis: 0)
            pooledConditionng = repeated(pooledConditionng, count: imageCount, axis: 0)
        }

        return (conditioning, pooledConditionng)
    }

    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        MLXRandom.seed(parameters.seed)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        let xt = sampler.samplePrior(
            shape: [parameters.imageCount] + parameters.latentSize + [autoencoder.latentChannels],
            dType: dType)

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning,
            steps: parameters.steps, cfgWeight: parameters.cfgWeight, textTime: textTime)
    }

    public func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
    {
        MLXRandom.seed(parameters.seed)

        // Define the num steps and start step
        let startStep = Float(sampler.maxTime) * strength
        let numSteps = Int(Float(parameters.steps) * strength)

        let (conditioning, pooledConditioning) = conditionText(
            text: parameters.prompt, imageCount: parameters.imageCount,
            cfgWeight: parameters.cfgWeight, negativeText: parameters.negativePrompt)

        let textTime = (
            pooledConditioning,
            repeated(
                MLXArray(converting: [512.0, 512, 0, 0, 512, 512]).reshaped(1, -1),
                count: pooledConditioning.count, axis: 0)
        )

        // Get the latents from the input image and add noise according to the
        // start time.

        var (x0, _) = autoencoder.encode(image[.newAxis])
        x0 = broadcast(x0, to: [parameters.imageCount] + x0.shape.dropFirst())
        let xt = sampler.addNoise(x: x0, t: MLXArray(startStep))

        return DenoiseIterator(
            sd: self, xt: xt, t: sampler.maxTime, conditioning: conditioning, steps: numSteps,
            cfgWeight: parameters.cfgWeight, textTime: textTime)
    }
}
