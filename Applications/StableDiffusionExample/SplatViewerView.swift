// Copyright © 2024 Apple Inc.

import SwiftUI
import UniformTypeIdentifiers

struct SplatViewerView: View {
    var splatStore: SplatStore

    @State private var manualURL: URL?
    @State private var showImporter = false
    @State private var rotation: Float = 0
    @State private var distance: Float = 8
    @State private var dragStart: Float = 0

    private var activeURL: URL? { manualURL ?? splatStore.combinedSplatURL }

    private var allowedContentTypes: [UTType] {
        ["ply", "splat", "spz"].compactMap { UTType(filenameExtension: $0) }
    }

    var body: some View {
        VStack(spacing: 8) {
            header
            Divider()
            content
        }
        .padding()
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: allowedContentTypes
        ) { result in
            if case .success(let url) = result {
                _ = url.startAccessingSecurityScopedResource()
                manualURL = url
            }
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 8) {
            if let url = activeURL {
                Image(systemName: "globe")
                Text(url.lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                if manualURL != nil {
                    Button("Use Generated") { manualURL = nil }
                        .font(.caption)
                        .disabled(splatStore.combinedSplatURL == nil)
                }
            } else {
                Text("No splat loaded")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                Spacer()
            }
            Button("Open File…") { showImporter = true }
                .font(.caption)
            Slider(value: $distance, in: 1...40)
                .frame(width: 140)
            Text("zoom")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var content: some View {
        if let url = activeURL {
            MetalKitSceneView(modelURL: url, rotation: $rotation, distance: $distance)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            if dragStart == 0 { dragStart = rotation }
                            rotation = dragStart + Float(value.translation.width) * 0.005
                        }
                        .onEnded { _ in dragStart = 0 }
                )
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else if splatStore.isBuilding {
            Spacer()
            ProgressView("Merging tiles into 360° splat...")
            Spacer()
        } else {
            Spacer()
            VStack(spacing: 10) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("Generate a panorama, crop it, and run SHARP on all tiles to build a 360° splat.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button("Open PLY / SPLAT / SPZ…") { showImporter = true }
                    .buttonStyle(.bordered)
            }
            Spacer()
        }
    }
}
