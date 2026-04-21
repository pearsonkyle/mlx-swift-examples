// Copyright © 2024 Apple Inc.

import CoreImage
import CoreML
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
    import AppKit
#else
    import UIKit
#endif

// MARK: - CroppedImageItem

struct CroppedImageItem: Identifiable {
    let id: Int
    let cgImage: CGImage

    var title: String { "Tile \(id + 1)" }
}

// MARK: - GaussianSplatEvaluator

@Observable @MainActor
class GaussianSplatEvaluator {
    var isRunning = false
    var progressText: String?
    var errorMessage: String?

    func run(
        cgImage: CGImage,
        modelPath: String,
        focalLength: Float,
        decimation: Float,
        outputURL: URL,
        completion: (@MainActor (Bool) -> Void)? = nil
    ) {
        isRunning = true
        errorMessage = nil
        progressText = "Initializing..."
        Task {
            await doRun(
                cgImage: cgImage,
                modelPath: modelPath,
                focalLength: focalLength,
                decimation: decimation,
                outputURL: outputURL
            )
            completion?(errorMessage == nil)
        }
    }

    /// Run SHARP on each item sequentially; manages isRunning state.
    /// Returns the output URLs that were successfully written.
    func runBatch(
        items: [(cgImage: CGImage, outputURL: URL)],
        modelPath: String,
        focalLength: Float,
        decimation: Float
    ) async -> [URL] {
        isRunning = true
        errorMessage = nil
        var results: [URL] = []
        for (cgImage, outputURL) in items {
            guard errorMessage == nil else { break }
            await doRunSingle(
                cgImage: cgImage,
                modelPath: modelPath,
                focalLength: focalLength,
                decimation: decimation,
                outputURL: outputURL
            )
            if errorMessage == nil {
                results.append(outputURL)
            }
        }
        isRunning = false
        progressText = nil
        return results
    }

    private func doRun(
        cgImage: CGImage,
        modelPath: String,
        focalLength: Float,
        decimation: Float,
        outputURL: URL
    ) async {
        await doRunSingle(
            cgImage: cgImage,
            modelPath: modelPath,
            focalLength: focalLength,
            decimation: decimation,
            outputURL: outputURL
        )
        progressText = nil
        isRunning = false
    }

    private func doRunSingle(
        cgImage: CGImage,
        modelPath: String,
        focalLength: Float,
        decimation: Float,
        outputURL: URL
    ) async {
        guard errorMessage == nil else { return }
        do {
            progressText = "Running SHARP inference..."
            try await Task.detached(priority: .userInitiated) {
                let runner = try SHARPModelRunner(modelPath: URL(filePath: modelPath))
                let imageArray = try runner.preprocessImage(cgImage: cgImage)
                let gaussians = try runner.predict(image: imageArray, focalLengthPx: focalLength)
                try runner.savePLY(
                    gaussians: gaussians,
                    focalLengthPx: focalLength,
                    imageShape: (height: runner.inputHeight, width: runner.inputWidth),
                    to: outputURL,
                    decimation: decimation
                )
            }.value
        } catch {
            errorMessage = "SHARP error: \(error.localizedDescription)"
        }
    }
}

// MARK: - CroppedImagesView

struct CroppedImagesView: View {

    let panoramaImage: CGImage?

    @State private var croppedItems: [CroppedImageItem] = []
    @State private var numTiles: Int = 4
    @State private var evaluator = GaussianSplatEvaluator()

    // SHARP settings
    @State private var sharpModelPath: String = ""
    @State private var focalLengthText: String = "1536"
    @State private var decimationText: String = "1.0"

    #if !os(macOS)
        @State private var documentPickerDelegate: DocumentPickerDelegate?
    #endif

    private let columns = [GridItem(.adaptive(minimum: 180, maximum: 280))]

    var body: some View {
        VStack(spacing: 10) {
            cropControlsBar
            sharpModelBar

            if evaluator.isRunning, let text = evaluator.progressText {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(text).font(.caption)
                }
                .padding(.vertical, 2)
            }

            if let msg = evaluator.errorMessage {
                Text(msg)
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            if croppedItems.isEmpty {
                emptyStateView
            } else {
                imagesGrid
            }
        }
        .padding()
        .onChange(of: panoramaImage) { _, _ in
            croppedItems = []
        }
    }

    // MARK: - Subviews

    @ViewBuilder
    var cropControlsBar: some View {
        HStack(spacing: 8) {
            Text("Tiles:")
                .foregroundStyle(.secondary)
            Stepper("\(numTiles)", value: $numTiles, in: 2...16)
                .frame(width: 100)
            Spacer()
            if !croppedItems.isEmpty {
                Button("Export All") { exportAll() }
                    .disabled(evaluator.isRunning)
                Button("Run SHARP on All") { runSHARPOnAll() }
                    .disabled(evaluator.isRunning || sharpModelPath.isEmpty)
            }
            Button("Crop Panorama") { cropPanorama() }
                .disabled(panoramaImage == nil)
                .buttonStyle(.borderedProminent)
        }
    }

    @ViewBuilder
    var sharpModelBar: some View {
        HStack(spacing: 6) {
            Text("SHARP Model:")
                .foregroundStyle(.secondary)
            TextField("path to .mlpackage", text: $sharpModelPath)
                .font(.caption)
                #if os(visionOS)
                    .textFieldStyle(.roundedBorder)
                #endif
            Button("Browse") { browseForSHARPModel() }
                .font(.caption)
            Divider().frame(height: 16)
            Text("Focal:")
                .foregroundStyle(.secondary)
            TextField("1536", text: $focalLengthText)
                .frame(width: 55)
                .font(.caption)
            Text("Decimate:")
                .foregroundStyle(.secondary)
            TextField("1.0", text: $decimationText)
                .frame(width: 40)
                .font(.caption)
        }
        .font(.caption)
    }

    @ViewBuilder
    var emptyStateView: some View {
        Spacer()
        VStack(spacing: 14) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            if panoramaImage == nil {
                Text("Generate a panorama first, then come here to crop and export tiles.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("Tap 'Crop Panorama' to slice the panorama into individual tiles.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button("Crop Panorama") { cropPanorama() }
                    .buttonStyle(.borderedProminent)
            }
        }
        Spacer()
    }

    @ViewBuilder
    var imagesGrid: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(croppedItems) { item in
                    CroppedImageCard(
                        item: item,
                        isRunning: evaluator.isRunning,
                        hasModel: !sharpModelPath.isEmpty,
                        onExport: { exportItem(item) },
                        onRunSHARP: { runSHARP(on: item) }
                    )
                }
            }
            .padding(.horizontal, 4)
        }
    }

    // MARK: - Crop Logic

    func cropPanorama() {
        guard let src = panoramaImage else { return }
        let tileWidth = src.width / numTiles
        guard tileWidth > 0 else { return }
        // Use a square aspect ratio: side = min(tileWidth, imageHeight),
        // centered vertically within the panorama.
        let side = min(tileWidth, src.height)
        let yOffset = (src.height - side) / 2
        croppedItems = (0..<numTiles).compactMap { i in
            let rect = CGRect(x: i * tileWidth, y: yOffset, width: side, height: side)
            guard let tile = src.cropping(to: rect) else { return nil }
            return CroppedImageItem(id: i, cgImage: tile)
        }
    }

    // MARK: - Export

    func exportItem(_ item: CroppedImageItem) {
        let filename = "panorama_tile_\(item.id + 1).png"
        #if os(macOS)
            let panel = NSSavePanel()
            panel.nameFieldStringValue = filename
            panel.allowedContentTypes = [.png]
            guard panel.runModal() == .OK, let url = panel.url else { return }
            try? writePNG(cgImage: item.cgImage, to: url)
        #else
            shareImage(item.cgImage, filename: filename)
        #endif
    }

    func exportAll() {
        guard !croppedItems.isEmpty else { return }
        #if os(macOS)
            let panel = NSOpenPanel()
            panel.title = "Choose Export Folder"
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.prompt = "Export Here"
            guard panel.runModal() == .OK, let folder = panel.url else { return }
            for item in croppedItems {
                let url = folder.appendingPathComponent("panorama_tile_\(item.id + 1).png")
                try? writePNG(cgImage: item.cgImage, to: url)
            }
        #else
            let images = croppedItems.map { UIImage(cgImage: $0.cgImage) }
            shareItems(images as [Any])
        #endif
    }

    // MARK: - SHARP Model Browser

    fileprivate static let sharpModelExtensions: Set<String> = ["mlpackage", "mlmodel", "mlmodelc"]

    func browseForSHARPModel() {
        #if os(macOS)
            let panel = NSOpenPanel()
            panel.title = "Select SHARP Core ML Model"
            // .mlpackage is a directory bundle without a registered UTType,
            // so allowedContentTypes won't match it. Instead we allow
            // directories and use a delegate to filter by extension.
            panel.canChooseDirectories = true
            panel.canChooseFiles = true
            panel.allowsMultipleSelection = false
            panel.treatsFilePackagesAsDirectories = false
            let panelDelegate = SHARPModelPanelDelegate()
            panel.delegate = panelDelegate
            if panel.runModal() == .OK, let url = panel.url {
                sharpModelPath = url.path(percentEncoded: false)
            }
        #else
            let types =
                ["mlpackage", "mlmodel", "mlmodelc"].compactMap { UTType(filenameExtension: $0) }
            let delegate = DocumentPickerDelegate { url in
                sharpModelPath = url.path(percentEncoded: false)
            }
            documentPickerDelegate = delegate
            let picker = UIDocumentPickerViewController(
                forOpeningContentTypes: types, asCopy: false)
            picker.allowsMultipleSelection = false
            picker.delegate = delegate
            presentViewController(picker)
        #endif
    }

    // MARK: - SHARP Inference

    func runSHARP(on item: CroppedImageItem) {
        let focalLength = Float(focalLengthText) ?? 1536.0
        let decimation = Float(decimationText) ?? 1.0
        let filename = "splat_tile_\(item.id + 1).ply"

        #if os(macOS)
            let panel = NSSavePanel()
            panel.nameFieldStringValue = filename
            if let plyType = UTType(filenameExtension: "ply") {
                panel.allowedContentTypes = [plyType]
            }
            guard panel.runModal() == .OK, let outputURL = panel.url else { return }
            evaluator.run(
                cgImage: item.cgImage,
                modelPath: sharpModelPath,
                focalLength: focalLength,
                decimation: decimation,
                outputURL: outputURL
            )
        #else
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(filename)
            evaluator.run(
                cgImage: item.cgImage,
                modelPath: sharpModelPath,
                focalLength: focalLength,
                decimation: decimation,
                outputURL: outputURL
            ) { success in
                if success { self.shareItems([outputURL]) }
            }
        #endif
    }

    func runSHARPOnAll() {
        guard !croppedItems.isEmpty else { return }
        let focalLength = Float(focalLengthText) ?? 1536.0
        let decimation = Float(decimationText) ?? 1.0

        #if os(macOS)
            let panel = NSOpenPanel()
            panel.title = "Choose Output Folder for PLY Files"
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.prompt = "Export Here"
            guard panel.runModal() == .OK, let folder = panel.url else { return }

            let batchItems = croppedItems.map { item in
                (
                    cgImage: item.cgImage,
                    outputURL: folder.appendingPathComponent("splat_tile_\(item.id + 1).ply")
                )
            }
            Task {
                await evaluator.runBatch(
                    items: batchItems,
                    modelPath: sharpModelPath,
                    focalLength: focalLength,
                    decimation: decimation
                )
            }
        #else
            let tmpDir = FileManager.default.temporaryDirectory
            let batchItems = croppedItems.map { item in
                (
                    cgImage: item.cgImage,
                    outputURL: tmpDir.appendingPathComponent("splat_tile_\(item.id + 1).ply")
                )
            }
            Task {
                let urls = await evaluator.runBatch(
                    items: batchItems,
                    modelPath: sharpModelPath,
                    focalLength: focalLength,
                    decimation: decimation
                )
                if !urls.isEmpty { shareItems(urls) }
            }
        #endif
    }

    // MARK: - Helpers

    func writePNG(cgImage: CGImage, to url: URL) throws {
        guard
            let dest = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            throw NSError(
                domain: "CroppedImagesView", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create image destination"])
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(
                domain: "CroppedImagesView", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to write PNG"])
        }
    }

    #if !os(macOS)
        func shareImage(_ cgImage: CGImage, filename: String) {
            shareItems([UIImage(cgImage: cgImage)])
        }

        func shareItems(_ items: [Any]) {
            let controller = UIActivityViewController(
                activityItems: items, applicationActivities: nil)
            presentViewController(controller)
        }

        func presentViewController(_ vc: UIViewController) {
            guard
                let windowScene = UIApplication.shared.connectedScenes
                    .compactMap({ $0 as? UIWindowScene }).first,
                let rootVC = windowScene.keyWindow?.rootViewController
            else { return }
            rootVC.present(vc, animated: true)
        }

        class DocumentPickerDelegate: NSObject, UIDocumentPickerDelegate {
            private let onFileSelected: (URL) -> Void

            init(onFileSelected: @escaping (URL) -> Void) {
                self.onFileSelected = onFileSelected
            }

            func documentPicker(
                _ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]
            ) {
                if let url = urls.first { onFileSelected(url) }
            }
        }
    #endif
}

// MARK: - SHARPModelPanelDelegate

#if os(macOS)
    /// Filters NSOpenPanel to only enable Core ML model bundles (.mlpackage, .mlmodel, .mlmodelc).
    class SHARPModelPanelDelegate: NSObject, NSOpenSavePanelDelegate {
        func panel(_ sender: Any, shouldEnable url: URL) -> Bool {
            // Always enable directories so the user can navigate into them,
            // but only allow selection of items with a matching extension.
            let ext = url.pathExtension.lowercased()
            if CroppedImagesView.sharpModelExtensions.contains(ext) {
                return true
            }
            // Enable plain directories for navigation
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue
            {
                return true
            }
            return false
        }
    }
#endif

// MARK: - CroppedImageCard

struct CroppedImageCard: View {
    let item: CroppedImageItem
    let isRunning: Bool
    let hasModel: Bool
    let onExport: () -> Void
    let onRunSHARP: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            Image(item.cgImage, scale: 1.0, label: Text(item.title))
                .resizable()
                .aspectRatio(contentMode: .fit)
                .cornerRadius(6)
                .shadow(radius: 2)

            Text(item.title)
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                Button("Export", action: onExport)
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .disabled(isRunning)

                Button("Run SHARP", action: onRunSHARP)
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .disabled(isRunning || !hasModel)
            }
        }
        .padding(8)
        .background(.regularMaterial)
        .cornerRadius(10)
    }
}
