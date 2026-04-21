// Copyright © 2024 Apple Inc.

import CoreGraphics
import Foundation

// MARK: - Core math (ported from haruishi43/equilib)

/// 3×3 matrix stored row-major
struct Mat3 {
    var m: [[Float]]  // 3×3

    static func identity() -> Mat3 {
        Mat3(m: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    }

    static func * (a: Mat3, b: Mat3) -> Mat3 {
        var r = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
        for i in 0..<3 {
            for j in 0..<3 {
                r[i][j] =
                    a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j]
            }
        }
        return Mat3(m: r)
    }

    func mulVec(_ v: (Float, Float, Float)) -> (Float, Float, Float) {
        (
            m[0][0] * v.0 + m[0][1] * v.1 + m[0][2] * v.2,
            m[1][0] * v.0 + m[1][1] * v.1 + m[1][2] * v.2,
            m[2][0] * v.0 + m[2][1] * v.1 + m[2][2] * v.2
        )
    }

    func inverted() -> Mat3 {
        let a = m
        let det =
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        let invDet = 1.0 / det
        var r = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
        r[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * invDet
        r[0][1] = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]) * invDet
        r[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * invDet
        r[1][0] = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]) * invDet
        r[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * invDet
        r[1][2] = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]) * invDet
        r[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * invDet
        r[2][1] = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]) * invDet
        r[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * invDet
        return Mat3(m: r)
    }
}

/// Intrinsic matrix from FOV and output size
private func intrinsicMatrix(width: Int, height: Int, fovXDeg: Float) -> Mat3 {
    let f = Float(width) / (2.0 * tan(fovXDeg * .pi / 360.0))
    return Mat3(m: [
        [f, 0, Float(width) / 2.0],
        [0, f, Float(height) / 2.0],
        [0, 0, 1],
    ])
}

/// Global-to-camera axis swap
private func global2CameraRotation() -> Mat3 {
    let rXY = Mat3(m: [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    let rYZ = Mat3(m: [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return rXY * rYZ
}

/// Rotation matrix from yaw/pitch/roll in radians
private func rotationMatrix(yaw: Float, pitch: Float, roll: Float) -> Mat3 {
    let (cr, sr) = (cos(roll), sin(roll))
    let (cp, sp) = (cos(pitch), sin(pitch))
    let (cy, sy) = (cos(yaw), sin(yaw))
    let rx = Mat3(m: [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    let ry = Mat3(m: [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    let rz = Mat3(m: [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz * ry * rx
}

// MARK: - Equi → Perspective extraction

/// Extract a perspective crop from a 2:1 equirectangular CGImage.
///
/// - Parameters:
///   - equiImage: Source equirectangular image (width = 2 × height).
///   - yawDeg, pitchDeg, rollDeg: Camera rotation in degrees.
///   - fovDeg: Horizontal field of view in degrees (e.g. 90).
///   - outWidth, outHeight: Output image size in pixels.
/// - Returns: The perspective crop as a CGImage.
func equi2pers(
    equiImage: CGImage,
    yawDeg: Float, pitchDeg: Float, rollDeg: Float,
    fovDeg: Float,
    outWidth: Int, outHeight: Int
) -> CGImage? {
    let wEqui = equiImage.width
    let hEqui = equiImage.height

    // Render source into a known RGBA8 format
    let srcBytesPerRow = wEqui * 4
    var srcPixels = [UInt8](repeating: 0, count: hEqui * srcBytesPerRow)
    guard
        let srcContext = CGContext(
            data: &srcPixels,
            width: wEqui, height: hEqui,
            bitsPerComponent: 8, bytesPerRow: srcBytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    else { return nil }
    srcContext.draw(
        equiImage, in: CGRect(x: 0, y: 0, width: wEqui, height: hEqui))

    // Precompute the combined transform
    let k = intrinsicMatrix(width: outWidth, height: outHeight, fovXDeg: fovDeg)
    let g2c = global2CameraRotation()
    let g = g2c * k.inverted()
    let r = rotationMatrix(
        yaw: yawDeg * .pi / 180,
        pitch: pitchDeg * .pi / 180,
        roll: rollDeg * .pi / 180
    )
    let c = r * g

    // Allocate output buffer
    let dstBytesPerRow = outWidth * 4
    var dstPixels = [UInt8](repeating: 0, count: outHeight * dstBytesPerRow)

    let wf = Float(wEqui)
    let hf = Float(hEqui)

    for py in 0..<outHeight {
        for px in 0..<outWidth {
            let vec = c.mulVec((Float(px), Float(py), 1.0))
            let norm = sqrt(vec.0 * vec.0 + vec.1 * vec.1 + vec.2 * vec.2)

            let phi = asin(vec.2 / norm)
            let theta = atan2(vec.1, vec.0)

            var equiX = (theta - .pi) * wf / (2.0 * .pi) + 0.5
            var equiY = (phi - .pi / 2.0) * hf / .pi + 0.5
            equiX = equiX.truncatingRemainder(dividingBy: wf)
            if equiX < 0 { equiX += wf }
            equiY = equiY.truncatingRemainder(dividingBy: hf)
            if equiY < 0 { equiY += hf }

            let x0 = Int(equiX)
            let x1 = (x0 + 1) % wEqui
            let y0 = Int(equiY)
            let y1 = (y0 + 1) % hEqui
            let fx = equiX - Float(x0)
            let fy = equiY - Float(y0)

            let dstOff = py * dstBytesPerRow + px * 4
            for ch in 0..<4 {
                let v00 = Float(srcPixels[y0 * srcBytesPerRow + x0 * 4 + ch])
                let v01 = Float(srcPixels[y0 * srcBytesPerRow + x1 * 4 + ch])
                let v10 = Float(srcPixels[y1 * srcBytesPerRow + x0 * 4 + ch])
                let v11 = Float(srcPixels[y1 * srcBytesPerRow + x1 * 4 + ch])
                let val =
                    v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy
                    + v11 * fx * fy
                dstPixels[dstOff + ch] = UInt8(min(max(val, 0), 255))
            }
        }
    }

    // Create output CGImage
    guard let provider = CGDataProvider(data: Data(dstPixels) as CFData),
        let outCG = CGImage(
            width: outWidth, height: outHeight,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: dstBytesPerRow,
            space: equiImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    else { return nil }

    return outCG
}

// MARK: - Evenly-spaced crops

/// Extract N perspective crops evenly distributed around the 360° equirectangular image.
///
/// - Parameters:
///   - equiImage: Source equirectangular CGImage (width = 2 × height).
///   - numCrops: Number of evenly-spaced views around the equator (default 4).
///   - fovDeg: Horizontal field of view of each crop in degrees (default 90).
///   - outputSize: Width and height of each square output image in pixels (default 512).
/// - Returns: An array of (yawDegrees, CGImage) pairs, or an empty array on failure.
func extractEvenlySpacedCrops(
    from equiImage: CGImage,
    numCrops: Int = 4,
    fovDeg: Float = 90,
    outputSize: Int = 512
) -> [(yaw: Float, image: CGImage)] {
    guard numCrops > 0 else { return [] }
    var results: [(yaw: Float, image: CGImage)] = []
    for i in 0..<numCrops {
        let yaw = Float(i) * 360.0 / Float(numCrops)
        if let crop = equi2pers(
            equiImage: equiImage,
            yawDeg: yaw, pitchDeg: 0, rollDeg: 0,
            fovDeg: fovDeg,
            outWidth: outputSize, outHeight: outputSize)
        {
            results.append((yaw: yaw, image: crop))
        }
    }
    return results
}
