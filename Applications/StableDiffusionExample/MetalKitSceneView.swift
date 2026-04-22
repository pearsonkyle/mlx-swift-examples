// Adapted from MetalSplatter's SampleApp by Scott Mielcarski
// https://github.com/scier/MetalSplatter
// Distributed under the MIT License.

#if os(iOS) || os(macOS)

import MetalKit
import SwiftUI

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif

struct MetalKitSceneView: ViewRepresentable {
    var modelURL: URL?
    @Binding var rotation: Float
    @Binding var distance: Float

    final class Coordinator {
        var renderer: MetalKitSceneRenderer?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    #if os(macOS)
    func makeNSView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateNSView(_ view: MTKView, context: Context) { updateView(context.coordinator) }
    #elseif os(iOS)
    func makeUIView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateUIView(_ view: MTKView, context: Context) { updateView(context.coordinator) }
    #endif

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()

        let renderer = MetalKitSceneRenderer(mtkView)
        coordinator.renderer = renderer
        mtkView.delegate = renderer
        renderer?.rotation = rotation
        renderer?.distance = distance

        if let url = modelURL {
            Task { try? await renderer?.load(url: url) }
        }
        return mtkView
    }

    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        renderer.rotation = rotation
        renderer.distance = distance
        Task { try? await renderer.load(url: modelURL) }
    }
}

#endif
