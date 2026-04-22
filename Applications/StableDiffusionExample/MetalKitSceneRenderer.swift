// Adapted from MetalSplatter's SampleApp by Scott Mielcarski
// https://github.com/scier/MetalSplatter
// Distributed under the MIT License.

#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import SplatIO
import os
import simd

@MainActor
final class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "StableDiffusionExample",
        category: "MetalKitSceneRenderer")

    private static let maxSimultaneousRenders = 3
    private static let fovyRadians: Float = 65 * .pi / 180

    let mtkView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var splatRenderer: SplatRenderer?
    private var loadedURL: URL?
    private var loadTask: Task<Void, Never>?

    private let inFlightSemaphore = DispatchSemaphore(value: MetalKitSceneRenderer.maxSimultaneousRenders)
    private var drawableSize: CGSize = .zero

    /// Yaw around +Y in radians. Driven by the SwiftUI view.
    var rotation: Float = 0
    /// Camera distance along -Z.
    var distance: Float = 8

    init?(_ mtkView: MTKView) {
        guard let device = mtkView.device,
              let queue = device.makeCommandQueue()
        else { return nil }
        self.device = device
        self.commandQueue = queue
        self.mtkView = mtkView
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.sampleCount = 1
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    }

    func load(url: URL?) async throws {
        if url == loadedURL { return }
        loadTask?.cancel()
        loadedURL = url
        splatRenderer = nil

        guard let url else { return }

        loadTask = Task { [device, mtkView] in
            do {
                let renderer = try SplatRenderer(
                    device: device,
                    colorFormat: mtkView.colorPixelFormat,
                    depthFormat: mtkView.depthStencilPixelFormat,
                    sampleCount: mtkView.sampleCount,
                    maxViewCount: 1,
                    maxSimultaneousRenders: Self.maxSimultaneousRenders)
                let reader = try AutodetectSceneReader(url)
                let points = try await reader.readAll()
                let chunk = try SplatChunk(device: device, from: points)
                await renderer.addChunk(chunk)
                await MainActor.run {
                    self.splatRenderer = renderer
                }
            } catch {
                Self.log.error("Failed to load splat at \(url.path): \(error.localizedDescription)")
            }
        }
    }

    private var viewport: SplatRenderer.ViewportDescriptor {
        let aspect = Float(max(drawableSize.width, 1) / max(drawableSize.height, 1))
        let projection = matrixPerspectiveRightHand(
            fovyRadians: Self.fovyRadians, aspectRatio: aspect, nearZ: 0.1, farZ: 200.0)
        let rot = matrix4x4Rotation(radians: rotation, axis: SIMD3<Float>(0, 1, 0))
        let trans = matrix4x4Translation(0, 0, -distance)
        // Common 3DGS PLYs expect +Y down, so roll 180° around Z to present rightside-up.
        let upFix = matrix4x4Rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let mtlViewport = MTLViewport(
            originX: 0, originY: 0,
            width: Double(drawableSize.width), height: Double(drawableSize.height),
            znear: 0, zfar: 1)

        return SplatRenderer.ViewportDescriptor(
            viewport: mtlViewport,
            projectionMatrix: projection,
            viewMatrix: trans * rot * upFix,
            screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    func draw(in view: MTKView) {
        guard let splatRenderer, splatRenderer.isReadyToRender else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: .distantFuture)
        guard let buffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }
        let semaphore = inFlightSemaphore
        buffer.addCompletedHandler { _ in semaphore.signal() }

        do {
            let viewport = self.viewport
            let didRender = try splatRenderer.render(
                viewports: [viewport],
                colorTexture: view.multisampleColorTexture ?? drawable.texture,
                colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: buffer)
            if didRender { buffer.present(drawable) }
        } catch {
            Self.log.error("Render failed: \(error.localizedDescription)")
        }

        buffer.commit()
    }
}

// MARK: - Minimal matrix helpers

private func matrix4x4Translation(_ tx: Float, _ ty: Float, _ tz: Float) -> simd_float4x4 {
    simd_float4x4(rows: [
        SIMD4<Float>(1, 0, 0, tx),
        SIMD4<Float>(0, 1, 0, ty),
        SIMD4<Float>(0, 0, 1, tz),
        SIMD4<Float>(0, 0, 0, 1),
    ])
}

private func matrix4x4Rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
    let a = simd_normalize(axis)
    let c = cos(radians)
    let s = sin(radians)
    let ci = 1 - c
    let x = a.x, y = a.y, z = a.z
    return simd_float4x4(rows: [
        SIMD4<Float>(c + x*x*ci,   x*y*ci - z*s, x*z*ci + y*s, 0),
        SIMD4<Float>(y*x*ci + z*s, c + y*y*ci,   y*z*ci - x*s, 0),
        SIMD4<Float>(z*x*ci - y*s, z*y*ci + x*s, c + z*z*ci,   0),
        SIMD4<Float>(0,            0,            0,            1),
    ])
}

private func matrixPerspectiveRightHand(
    fovyRadians: Float, aspectRatio: Float, nearZ: Float, farZ: Float
) -> simd_float4x4 {
    let ys = 1 / tan(fovyRadians * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return simd_float4x4(rows: [
        SIMD4<Float>(xs, 0,  0,  0),
        SIMD4<Float>(0,  ys, 0,  0),
        SIMD4<Float>(0,  0,  zs, zs * nearZ),
        SIMD4<Float>(0,  0,  -1, 0),
    ])
}

#endif
