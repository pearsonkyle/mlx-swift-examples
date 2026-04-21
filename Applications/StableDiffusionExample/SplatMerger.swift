// Copyright © 2024 Apple Inc.
//
// Combines per-tile SHARP gaussian splats into a single 360° scene by
// rotating each tile around the world Y axis by its angular slot.

import CoreML
import Foundation

enum SplatMerger {

    /// Build a merged PLY where tile `i` of `gaussians.count` is rotated by
    /// `(i + 0.5) · 2π / N` radians around +Y. Mirrors the attribute layout
    /// produced by `SHARPModelRunner.savePLY` so MetalSplatter's SplatIO
    /// reader treats it identically.
    ///
    /// `decimation` is a keep-ratio in `(0, 1]`; values below 1 run the same
    /// importance-weighted selection used by `SHARPModelRunner.savePLY`, so
    /// the merged file stays consistent with the per-tile PLYs written beside it.
    static func mergePanorama360(
        gaussians: [Gaussians3D],
        focalLengthPx: Float,
        imageShape: (height: Int, width: Int),
        decimation: Float = 1.0,
        to outputURL: URL
    ) throws {
        let n = gaussians.count
        guard n > 0 else {
            throw NSError(
                domain: "SplatMerger", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No tiles to merge"])
        }

        let clampedDecimation = min(max(decimation, 0.0001), 1.0)
        let keepIndicesPerTile: [[Int]] = gaussians.map { tile in
            clampedDecimation < 1.0
                ? tile.decimationIndices(keepRatio: clampedDecimation)
                : Array(0..<tile.count)
        }
        let totalCount = keepIndicesPerTile.reduce(0) { $0 + $1.count }
        let imageWidth = imageShape.width
        let imageHeight = imageShape.height

        var data = Data()

        func appendString(_ s: String) { data.append(s.data(using: .ascii)!) }
        func appendFloat(_ v: Float) {
            var x = v
            data.append(Data(bytes: &x, count: 4))
        }
        func appendInt32(_ v: Int32) {
            var x = v
            data.append(Data(bytes: &x, count: 4))
        }
        func appendUInt32(_ v: UInt32) {
            var x = v
            data.append(Data(bytes: &x, count: 4))
        }
        func appendUInt8(_ v: UInt8) {
            var x = v
            data.append(Data(bytes: &x, count: 1))
        }

        // ===== Header (matches Sharp.savePLY) =====
        appendString("ply\n")
        appendString("format binary_little_endian 1.0\n")
        appendString("element vertex \(totalCount)\n")
        appendString("property float x\n")
        appendString("property float y\n")
        appendString("property float z\n")
        appendString("property float f_dc_0\n")
        appendString("property float f_dc_1\n")
        appendString("property float f_dc_2\n")
        appendString("property float opacity\n")
        appendString("property float scale_0\n")
        appendString("property float scale_1\n")
        appendString("property float scale_2\n")
        appendString("property float rot_0\n")
        appendString("property float rot_1\n")
        appendString("property float rot_2\n")
        appendString("property float rot_3\n")
        appendString("element extrinsic 16\n")
        appendString("property float extrinsic\n")
        appendString("element intrinsic 9\n")
        appendString("property float intrinsic\n")
        appendString("element image_size 2\n")
        appendString("property uint image_size\n")
        appendString("element frame 2\n")
        appendString("property int frame\n")
        appendString("element disparity 2\n")
        appendString("property float disparity\n")
        appendString("element color_space 1\n")
        appendString("property uchar color_space\n")
        appendString("element version 3\n")
        appendString("property uchar version\n")
        appendString("end_header\n")

        // ===== Vertex data =====
        var disparities: [Float] = []

        for (i, tile) in gaussians.enumerated() {
            // Tile-center azimuth in a cylinder divided into N wedges.
            let theta = (Float(i) + 0.5) * 2.0 * .pi / Float(n)
            let c = cos(theta)
            let s = sin(theta)
            // Compose world rotation q_y(θ) * q_local (left multiply).
            let halfTheta = theta * 0.5
            let qyw = cos(halfTheta)
            let qyy = sin(halfTheta)

            let meanPtr = tile.meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
            let scalePtr = tile.singularValues.dataPointer.assumingMemoryBound(to: Float.self)
            let quatPtr = tile.quaternions.dataPointer.assumingMemoryBound(to: Float.self)
            let colorPtr = tile.colors.dataPointer.assumingMemoryBound(to: Float.self)
            let opacityPtr = tile.opacities.dataPointer.assumingMemoryBound(to: Float.self)

            for j in keepIndicesPerTile[i] {
                // Position: rotate around +Y by θ.
                let x0 = meanPtr[j * 3 + 0]
                let y0 = meanPtr[j * 3 + 1]
                let z0 = meanPtr[j * 3 + 2]
                let x = c * x0 + s * z0
                let y = y0
                let z = -s * x0 + c * z0
                appendFloat(x)
                appendFloat(y)
                appendFloat(z)

                if z > 1e-6 {
                    disparities.append(1.0 / z)
                }

                // Colors: linearRGB -> sRGB -> SH degree-0.
                let r = colorPtr[j * 3 + 0]
                let g = colorPtr[j * 3 + 1]
                let b = colorPtr[j * 3 + 2]
                appendFloat(rgbToSphericalHarmonics(linearRGBToSRGB(r)))
                appendFloat(rgbToSphericalHarmonics(linearRGBToSRGB(g)))
                appendFloat(rgbToSphericalHarmonics(linearRGBToSRGB(b)))

                // Opacity: inverse sigmoid.
                appendFloat(inverseSigmoid(opacityPtr[j]))

                // Scales: log-space.
                appendFloat(log(max(scalePtr[j * 3 + 0], 1e-10)))
                appendFloat(log(max(scalePtr[j * 3 + 1], 1e-10)))
                appendFloat(log(max(scalePtr[j * 3 + 2], 1e-10)))

                // Rotation: left-multiply q_y(θ) * q_local. Quaternion order is (w, x, y, z).
                let qw = quatPtr[j * 4 + 0]
                let qx = quatPtr[j * 4 + 1]
                let qy = quatPtr[j * 4 + 2]
                let qz = quatPtr[j * 4 + 3]
                // Hamilton product with q_y = (qyw, 0, qyy, 0):
                //   r.w = qyw·qw - qyy·qy
                //   r.x = qyw·qx - qyy·qz    (since q_y.x=0, q_y.z=0)
                //   r.y = qyw·qy + qyy·qw
                //   r.z = qyw·qz + qyy·qx
                var rw = qyw * qw - qyy * qy
                var rx = qyw * qx - qyy * qz
                var ry = qyw * qy + qyy * qw
                var rz = qyw * qz + qyy * qx
                // Renormalize to guard against drift.
                let invN = 1.0 / max(sqrt(rw * rw + rx * rx + ry * ry + rz * rz), 1e-10)
                rw *= invN; rx *= invN; ry *= invN; rz *= invN
                appendFloat(rw)
                appendFloat(rx)
                appendFloat(ry)
                appendFloat(rz)
            }
        }

        // ===== Extrinsic (identity 4x4) =====
        let identity: [Float] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]
        for v in identity { appendFloat(v) }

        // ===== Intrinsic (per-tile focal/principal point; tiles share these) =====
        let intrinsic: [Float] = [
            focalLengthPx, 0, Float(imageWidth) * 0.5,
            0, focalLengthPx, Float(imageHeight) * 0.5,
            0, 0, 1,
        ]
        for v in intrinsic { appendFloat(v) }

        // ===== Image size =====
        appendUInt32(UInt32(imageWidth))
        appendUInt32(UInt32(imageHeight))

        // ===== Frame (1 frame, totalCount particles) =====
        appendInt32(1)
        appendInt32(Int32(totalCount))

        // ===== Disparity quantiles =====
        disparities.sort()
        let d10 = disparities.isEmpty
            ? 0.0 : disparities[min(Int(Float(disparities.count) * 0.1), disparities.count - 1)]
        let d90 = disparities.isEmpty
            ? 1.0 : disparities[min(Int(Float(disparities.count) * 0.9), disparities.count - 1)]
        appendFloat(d10)
        appendFloat(d90)

        // ===== Color space = sRGB =====
        appendUInt8(1)

        // ===== Version (major, minor, patch) — matches Sharp.savePLY =====
        appendUInt8(1)
        appendUInt8(5)
        appendUInt8(0)

        try data.write(to: outputURL)
    }
}
