// Copyright © 2024 Apple Inc.

import MLX
import StableDiffusion
import SwiftUI
import UniformTypeIdentifiers
#if !os(macOS)
import UIKit
#endif
// MARK: - Model Source

/// Selects between the default SDXL Turbo preset and a custom local model.
enum ModelSource: String, CaseIterable {
    case sdxlTurbo = "SDXL Turbo"
    case localModel = "Local Model (Panorama)"
}

struct ContentView: View {

    @State var prompt = "Glowing mushrooms around pyramids amidst a cosmic backdrop, equirectangular, 360 panorama, cinematic"
    @State var negativePrompt =
        "boring, text, signature, watermark, low quality, bad quality, grainy, blurry, long neck, closed eyes"
    @State var evaluator = StableDiffusionEvaluator()
    @State var showProgress = false

    // Local model file path (single merged checkpoint with LoRA baked in)
    @State var modelSource: ModelSource = .sdxlTurbo
    @State var checkpointPath: String = ""
    #if !os(macOS)
    @State var documentPickerDelegate: DocumentPickerDelegate?
    #endif

    // Panorama parameters
    @State var outputWidth: String = "2048"
    @State var outputHeight: String = "1024"
    @State var steps: String = "8"
    @State var cfgScale: String = "3.0"
    @State var seed: String = ""

    var body: some View {
        TabView {
            generationTab
                .tabItem {
                    Label("Generate", systemImage: "wand.and.stars")
                }
            CroppedImagesView(panoramaImage: evaluator.image)
                .tabItem {
                    Label("Gaussian Splat", systemImage: "cube.transparent")
                }
        }
    }

    var generationTab: some View {
        VStack {
            VStack(spacing: 4) {
                if let progress = evaluator.progress {
                    if progress.isIndeterminate {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text(progress.title)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        ProgressView(value: progress.current, total: progress.limit) {
                            HStack {
                                Text(progress.title)
                                    .font(.caption)
                                Spacer()
                                Text("\(Int(progress.current))/\(Int(progress.limit))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    if let detail = progress.detail {
                        Text(detail)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 40)

            Spacer()
            if let image = evaluator.image {
                Image(image, scale: 1.0, label: Text(""))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minHeight: 200)
            }
            Spacer()

            // Model Source Picker
            Picker("Model", selection: $modelSource) {
                ForEach(ModelSource.allCases, id: \.self) { source in
                    Text(source.rawValue).tag(source)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            if modelSource == .localModel {
                localModelSection
            }

            Grid {
                GridRow {
                    TextField("prompt", text: $prompt)
                        .onSubmit(generate)
                        .disabled(evaluator.progress != nil)
                        #if os(visionOS)
                            .textFieldStyle(.roundedBorder)
                        #endif

                    Button(action: { prompt = "" }) {
                        Label("clear", systemImage: "xmark.circle.fill").font(.system(size: 10))
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)

                    Button("generate", action: generate)
                        .disabled(evaluator.progress != nil)
                        .keyboardShortcut("r")
                }

                GridRow {
                    TextField("negative prompt", text: $negativePrompt)
                        .onSubmit(generate)
                        .disabled(evaluator.progress != nil)
                        #if os(visionOS)
                            .textFieldStyle(.roundedBorder)
                        #endif
                    Button(action: { negativePrompt = "" }) {
                        Label("clear", systemImage: "xmark.circle.fill").font(
                            .system(size: 10))
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.plain)

                    if modelSource == .sdxlTurbo {
                        Toggle("Show Progress", isOn: $showProgress)
                    } else {
                        EmptyView()
                    }
                }

                if modelSource == .localModel {
                    GridRow {
                        HStack {
                            Text("Size:")
                                .foregroundStyle(.secondary)
                            TextField("W", text: $outputWidth)
                                .frame(width: 60)
                            Text("×")
                            TextField("H", text: $outputHeight)
                                .frame(width: 60)
                            Text("Steps:")
                                .foregroundStyle(.secondary)
                            TextField("Steps", text: $steps)
                                .frame(width: 40)
                            Text("CFG:")
                                .foregroundStyle(.secondary)
                            TextField("CFG", text: $cfgScale)
                                .frame(width: 40)
                            Text("Seed:")
                                .foregroundStyle(.secondary)
                            TextField("random", text: $seed)
                                .frame(width: 80)
                        }
                    }
                }
            }
            .frame(minWidth: 300)

            if let message = evaluator.message {
                Text(message)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .padding()
    }

    // MARK: - Local Model File Selection

    @ViewBuilder
    var localModelSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            fileRow(label: "Merged Checkpoint:", path: $checkpointPath, types: ["safetensors"])
        }
        .padding(.horizontal)
        .font(.caption)
    }

    func fileRow(label: String, path: Binding<String>, types: [String]) -> some View {
        HStack {
            Text(label)
                .frame(width: 130, alignment: .trailing)
                .foregroundStyle(.secondary)
            TextField("path to .safetensors", text: path)
                .font(.caption)
                #if os(visionOS)
                    .textFieldStyle(.roundedBorder)
                #endif
            Button("Browse") {
                #if os(macOS)
                let panel = NSOpenPanel()
                panel.allowedContentTypes = types.compactMap { UTType(filenameExtension: $0) }
                panel.canChooseDirectories = false
                panel.allowsMultipleSelection = false
                if panel.runModal() == .OK, let url = panel.url {
                    path.wrappedValue = url.path(percentEncoded: false)
                }
                #else
                let delegate = DocumentPickerDelegate(
                    onFileSelected: { url in
                        path.wrappedValue = url.path(percentEncoded: false)
                    }
                )
                documentPickerDelegate = delegate

                let documentPicker = UIDocumentPickerViewController(
                    forOpeningContentTypes: types.compactMap { UTType(filenameExtension: $0) },
                    asCopy: true
                )
                documentPicker.allowsMultipleSelection = false
                documentPicker.directoryURL = URL(filePath: path.wrappedValue)
                documentPicker.delegate = delegate

                if let windowScene = UIApplication.shared.connectedScenes
                    .compactMap({ $0 as? UIWindowScene }).first,
                   let rootVC = windowScene.keyWindow?.rootViewController {
                    rootVC.present(documentPicker, animated: true)
                }
                #endif
            }
            .font(.caption)
        }
    }

    // Document picker delegate for iOS/visionOS
    #if !os(macOS)
    class DocumentPickerDelegate: NSObject, UIDocumentPickerDelegate {
        private let onFileSelected: (URL) -> Void
        
        init(onFileSelected: @escaping (URL) -> Void) {
            self.onFileSelected = onFileSelected
        }
        
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first {
                onFileSelected(url)
            }
        }
    }
    #endif
    // MARK: - Generation

    private func generate() {
        switch modelSource {
        case .sdxlTurbo:
            Task {
                await evaluator.generate(
                    prompt: prompt, negativePrompt: negativePrompt, showProgress: showProgress)
            }
        case .localModel:
            Task {
                let w = Int(outputWidth) ?? 2048
                let h = Int(outputHeight) ?? 1024
                let s = Int(steps) ?? 8
                let cfg = Float(cfgScale) ?? 3.0
                let seedVal: UInt64? = seed.isEmpty ? nil : UInt64(seed)

                await evaluator.generatePanorama(
                    prompt: prompt,
                    negativePrompt: negativePrompt,
                    checkpointPath: checkpointPath,
                    width: w,
                    height: h,
                    steps: s,
                    cfgScale: cfg,
                    seed: seedVal
                )
            }
        }
    }
}

/// Progress reporting with a title, elapsed time, and optional indeterminate mode.
struct Progress: Equatable {
    let title: String
    let current: Double
    let limit: Double
    /// When true, shows an indeterminate (animated) indicator instead of a bar
    let isIndeterminate: Bool
    /// Detail text shown below the title (e.g. "Step 3/8 — 12.4s per step")
    let detail: String?

    init(title: String, current: Double, limit: Double, isIndeterminate: Bool = false, detail: String? = nil) {
        self.title = title
        self.current = current
        self.limit = limit
        self.isIndeterminate = isIndeterminate
        self.detail = detail
    }
}

/// Async model factory
actor ModelFactory {

    enum LoadState {
        case idle
        case loading(Task<ModelContainer<TextToImageGenerator>, Error>)
        case loaded(ModelContainer<TextToImageGenerator>)
    }

    enum SDError: LocalizedError {
        case unableToLoad

        var errorDescription: String? {
            switch self {
            case .unableToLoad:
                return String(
                    localized:
                        "Unable to load the Stable Diffusion model. Please check your internet connection or available storage space."
                )
            }
        }
    }

    public nonisolated let configuration = StableDiffusionConfiguration.presetSDXLTurbo

    /// if true we show UI that lets users see the intermediate steps
    public nonisolated let canShowProgress: Bool

    /// if true we show UI to give negative text
    public nonisolated let canUseNegativeText: Bool

    private var loadState = LoadState.idle
    private var loadConfiguration = LoadConfiguration(float16: true, quantize: false)

    public nonisolated let conserveMemory: Bool

    init() {
        let defaultParameters = configuration.defaultParameters()
        self.canShowProgress = defaultParameters.steps > 4
        self.canUseNegativeText = defaultParameters.cfgWeight > 1

        // this will be true e.g. if the computer has 8G of memory or less
        self.conserveMemory = Memory.memoryLimit < 8 * 1024 * 1024 * 1024

        if conserveMemory {
            print("conserving memory")
            loadConfiguration.quantize = true
            Memory.cacheLimit = 1 * 1024 * 1024
            Memory.memoryLimit = 3 * 1024 * 1024 * 1024
        } else {
            Memory.cacheLimit = 256 * 1024 * 1024
        }
    }

    public func load(reportProgress: @escaping @Sendable (Progress) -> Void) async throws
        -> ModelContainer<TextToImageGenerator>
    {
        switch loadState {
        case .idle:
            let task = Task {
                do {
                    try await configuration.download { progress in
                        if progress.fractionCompleted < 0.99 {
                            reportProgress(
                                .init(
                                    title: "Download", current: progress.fractionCompleted * 100,
                                    limit: 100))
                        }
                    }
                } catch {
                    let nserror = error as NSError
                    if nserror.domain == NSURLErrorDomain
                        && nserror.code == NSURLErrorNotConnectedToInternet
                    {
                        // Internet connection appears to be offline -- fall back to loading from
                        // the local directory
                        reportProgress(.init(title: "Offline", current: 100, limit: 100))
                    } else {
                        throw error
                    }
                }

                let container = try ModelContainer<TextToImageGenerator>.createTextToImageGenerator(
                    configuration: configuration, loadConfiguration: loadConfiguration)

                await container.setConserveMemory(conserveMemory)

                try await container.perform { model in
                    reportProgress(.init(title: "Loading weights", current: 0, limit: 1))
                    if !conserveMemory {
                        model.ensureLoaded()
                    }
                }

                return container
            }
            self.loadState = .loading(task)

            let container = try await task.value

            if conserveMemory {
                // if conserving memory return the model but do not keep it in memory
                self.loadState = .idle
            } else {
                // cache the model in memory to make it faster to run with new prompts
                self.loadState = .loaded(container)
            }

            return container

        case .loading(let task):
            let generator = try await task.value
            return generator

        case .loaded(let generator):
            return generator
        }
    }

}

@Observable @MainActor
class StableDiffusionEvaluator {

    var progress: Progress?
    var message: String?
    var image: CGImage?

    let modelFactory = ModelFactory()

    @Sendable
    nonisolated private func updateProgress(progress: Progress?) {
        Task { @MainActor in
            self.progress = progress
        }
    }

    @Sendable
    nonisolated private func updateImage(image: CGImage?) {
        Task { @MainActor in
            self.image = image
        }
    }

    nonisolated private func display(decoded: MLXArray) {
        let raster = (decoded * 255).asType(.uint8).squeezed()
        let image = Image(raster).asCGImage()

        Task { @MainActor in
            updateImage(image: image)
        }
    }

    // MARK: - SDXL Turbo Generation (Original)

    func generate(prompt: String, negativePrompt: String, showProgress: Bool) async {
        progress = .init(title: "Preparing", current: 0, limit: 1)
        message = nil

        let parameters = {
            var p = modelFactory.configuration.defaultParameters()
            p.prompt = prompt
            p.negativePrompt = negativePrompt

            if modelFactory.conserveMemory {
                p.steps = 1
            }

            return p
        }()

        do {
            let container = try await modelFactory.load(reportProgress: updateProgress)

            try await container.performTwoStage { generator in
                var parameters = modelFactory.configuration.defaultParameters()
                parameters.prompt = prompt
                parameters.negativePrompt = negativePrompt

                if modelFactory.conserveMemory {
                    parameters.steps = 1
                }

                let latents: DenoiseIterator? = generator.generateLatents(parameters: parameters)

                return (generator.detachedDecoder(), latents)

            } second: { decoder, latents in
                var lastXt: MLXArray?
                for (i, xt) in latents!.enumerated() {
                    lastXt = nil
                    eval(xt)
                    lastXt = xt

                    if showProgress, i % 10 == 0 {
                        display(decoded: decoder(xt))
                    }

                    updateProgress(
                        progress: .init(
                            title: "Generate Latents", current: Double(i),
                            limit: Double(parameters.steps)))
                }

                if let lastXt {
                    display(decoded: decoder(lastXt))
                }
                updateProgress(progress: nil)
            }

        } catch {
            progress = nil
            message = "Failed: \(error)"
        }
    }

    // MARK: - Local Model Panorama Generation (Merged Checkpoint)

    func generatePanorama(
        prompt: String,
        negativePrompt: String,
        checkpointPath: String,
        width: Int,
        height: Int,
        steps: Int,
        cfgScale: Float,
        seed: UInt64?
    ) async {
        progress = .init(title: "Preparing local model...", current: 0, limit: 1, isIndeterminate: true)
        message = nil

        let overallStart = CFAbsoluteTimeGetCurrent()

        do {
            // Validate checkpoint path
            guard !checkpointPath.isEmpty else {
                message = "Please select a merged checkpoint file"
                progress = nil
                return
            }

            let checkpointUrl = URL(filePath: checkpointPath)

            // Step 1: Ensure SDXL Turbo is downloaded (for tokenizer/scheduler files)
            updateProgress(
                progress: .init(
                    title: "Downloading tokenizer files",
                    current: 0, limit: 100,
                    detail: "Required for SDXL tokenizer and scheduler"
                )
            )
            let sdxlTurbo = StableDiffusionConfiguration.presetSDXLTurbo
            let progressCallback: @Sendable (Foundation.Progress) -> Void = { [weak self] dlProgress in
                if dlProgress.fractionCompleted < 0.99 {
                    self?.updateProgress(
                        progress: .init(
                            title: "Downloading tokenizer files",
                            current: dlProgress.fractionCompleted * 100,
                            limit: 100,
                            detail: String(format: "%.0f%% complete", dlProgress.fractionCompleted * 100)
                        )
                    )
                }
            }
            try await sdxlTurbo.download(progressHandler: progressCallback)

            // Step 2: Load the merged checkpoint (VAE + LoRA already baked in)
            let loadStart = CFAbsoluteTimeGetCurrent()
            updateProgress(
                progress: .init(
                    title: "Loading checkpoint",
                    current: 0, limit: 1,
                    isIndeterminate: true,
                    detail: "Parsing \(checkpointUrl.lastPathComponent)..."
                )
            )

            let sd = try loadStableDiffusionXLFromSingleFile(
                url: checkpointUrl,
                dType: LoadConfiguration().dType
            )

            let loadElapsed = CFAbsoluteTimeGetCurrent() - loadStart

            // Step 3: Create panorama generator and load weights into GPU
            let weightsStart = CFAbsoluteTimeGetCurrent()
            updateProgress(
                progress: .init(
                    title: "Loading model weights",
                    current: 0, limit: 1,
                    isIndeterminate: true,
                    detail: String(format: "Checkpoint loaded in %.1fs — transferring weights to GPU...", loadElapsed)
                )
            )
            let generator = PanoramaGenerator(sd, width: width, height: height)
            generator.ensureLoaded()

            let weightsElapsed = CFAbsoluteTimeGetCurrent() - weightsStart

            // Step 4: Generate panorama
            let parameters = PanoramaParameters(
                prompt: prompt,
                negativePrompt: negativePrompt,
                width: width,
                height: height,
                steps: steps,
                cfgScale: cfgScale,
                seed: seed
            )

            let decoder = generator.detachedDecoder()
            let latents = generator.generateLatents(parameters: parameters)

            updateProgress(
                progress: .init(
                    title: "Denoising",
                    current: 0, limit: Double(steps),
                    detail: String(format: "Weights loaded in %.1fs — starting denoising (%d×%d, %d steps)...", weightsElapsed, width, height, steps)
                )
            )

            var lastXt: MLXArray?
            var stepTimes: [Double] = []
            for (i, xt) in latents.enumerated() {
                let stepStart = CFAbsoluteTimeGetCurrent()
                lastXt = nil
                eval(xt)
                lastXt = xt

                let stepElapsed = CFAbsoluteTimeGetCurrent() - stepStart
                stepTimes.append(stepElapsed)

                let avgStepTime = stepTimes.reduce(0, +) / Double(stepTimes.count)
                let remainingSteps = steps - (i + 1)
                let eta = avgStepTime * Double(remainingSteps)

                let detail: String
                if remainingSteps > 0 {
                    detail = String(format: "Step %d/%d (%.1fs) — ~%.0fs remaining", i + 1, steps, stepElapsed, eta)
                } else {
                    let totalDenoise = stepTimes.reduce(0, +)
                    detail = String(format: "Step %d/%d (%.1fs) — denoising done in %.1fs", i + 1, steps, stepElapsed, totalDenoise)
                }

                updateProgress(
                    progress: .init(
                        title: "Denoising",
                        current: Double(i + 1),
                        limit: Double(steps),
                        detail: detail
                    )
                )
            }

            // Step 5: Decode and display
            if let lastXt {
                let decodeStart = CFAbsoluteTimeGetCurrent()
                updateProgress(
                    progress: .init(
                        title: "Decoding image",
                        current: 0, limit: 1,
                        isIndeterminate: true,
                        detail: "Running VAE decoder..."
                    )
                )
                display(decoded: decoder(lastXt))
                let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
                let totalElapsed = CFAbsoluteTimeGetCurrent() - overallStart
                updateProgress(
                    progress: .init(
                        title: "Complete",
                        current: 1, limit: 1,
                        detail: String(format: "Decoded in %.1fs — total time %.1fs", decodeElapsed, totalElapsed)
                    )
                )
                // Brief pause so the user can see the completion message
                try await Task.sleep(for: .seconds(1.5))
            }

            updateProgress(progress: nil)

        } catch {
            progress = nil
            message = "Failed: \(error.localizedDescription)"
        }
    }
}
