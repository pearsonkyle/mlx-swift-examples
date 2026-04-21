//
//  SHARPModelRunner.swift
//  SHARP Model Inference and PLY Export
//
//  Loads a SHARP Core ML model, runs inference on an image,
//  and saves the 3D Gaussian splat output as a PLY file.
// 
//  Usage:
//    swiftc -O -o sharp_runner sharp.swift -framework CoreML -framework CoreImage -framework AppKit
//    ./sharp_runner sharp.mlpackage test.png output.ply -d 0.5

import Foundation
import CoreML
import CoreImage
import AppKit  // For NSImage on macOS; use UIKit for iOS

// MARK: - Gaussians3D Structure

/// Represents the output of the SHARP model - a collection of 3D Gaussians
struct Gaussians3D {
    let meanVectors: MLMultiArray      // Shape: (1, N, 3) - 3D positions
    let singularValues: MLMultiArray   // Shape: (1, N, 3) - scales
    let quaternions: MLMultiArray      // Shape: (1, N, 4) - rotations
    let colors: MLMultiArray           // Shape: (1, N, 3) - RGB colors (linear)
    let opacities: MLMultiArray        // Shape:  (1, N) - opacity values
    
    var count: Int {
        return meanVectors.shape[1].intValue
    }
    
    /// Compute importance scores for each Gaussian.
    /// Higher scores = more important (larger and more opaque).
    func computeImportanceScores() -> [Float] {
        let n = count
        var scores = [Float](repeating: 0, count: n)
        
        let scalePtr = singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = opacities.dataPointer.assumingMemoryBound(to: Float.self)
        
        for i in 0..<n {
            // Sum of log scales (singular values are already in linear space, not log)
            // To match Python:  scales = exp(scale_0 + scale_1 + scale_2)
            // But our singularValues are already exp(log_scale), so we need log them first
            let s0 = scalePtr[i * 3 + 0]
            let s1 = scalePtr[i * 3 + 1]
            let s2 = scalePtr[i * 3 + 2]
            
            // Product of scales (equivalent to exp(log_s0 + log_s1 + log_s2))
            let scaleProduct = s0 * s1 * s2
            
            // Opacity is already in [0, 1] range (after sigmoid in model)
            let opacity = opacityPtr[i]
            
            scores[i] = scaleProduct * opacity
        }
        
        return scores
    }
    
    /// Decimate the Gaussians by keeping only a fraction based on importance.
    /// Returns indices of Gaussians to keep, sorted for spatial coherence.
    func decimationIndices(keepRatio: Float) -> [Int] {
        let n = count
        let keepCount = max(1, Int(Float(n) * keepRatio))
        
        // Compute importance scores
        let scores = computeImportanceScores()
        
        // Create array of (index, score) pairs and sort by score descending
        var indexedScores = scores.enumerated().map { ($0.offset, $0.element) }
        indexedScores.sort { $0.1 > $1.1 }
        
        // Get top keepCount indices
        var keepIndices = indexedScores.prefix(keepCount).map { $0.0 }
        
        // Sort indices to maintain spatial coherence
        keepIndices.sort()
        
        return keepIndices
    }
}

// MARK: - Color Space Utilities

/// Convert linear RGB to sRGB color space
func linearRGBToSRGB(_ linear: Float) -> Float {
    if linear <= 0.0031308 {
        return linear * 12.92
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055
    }
}

/// Convert RGB to degree-0 spherical harmonics
func rgbToSphericalHarmonics(_ rgb: Float) -> Float {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return (rgb - 0.5) / coeffDegree0
}

/// Inverse sigmoid function
func inverseSigmoid(_ x: Float) -> Float {
    let clamped = min(max(x, 1e-6), 1.0 - 1e-6)
    return log(clamped / (1.0 - clamped))
}

// MARK: - SHARP Model Wrapper

class SHARPModelRunner {
    private let model: MLModel
    private let inputHeight: Int
    private let inputWidth: Int
    
    init(modelPath: URL, inputHeight: Int = 1536, inputWidth: Int = 1536) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        // Compile the model if needed
        let compiledModelURL = try SHARPModelRunner.compileModelIfNeeded(at: modelPath)
        
        self.model = try MLModel(contentsOf: compiledModelURL, configuration:  config)
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        
        // Print model description for debugging
        print("Model inputs: \(model.modelDescription.inputDescriptionsByName.keys.joined(separator: ", "))")
        print("Model outputs:  \(model.modelDescription.outputDescriptionsByName.keys.joined(separator: ", "))")
    }
    
    /// Compile the model if it's not already compiled
    private static func compileModelIfNeeded(at modelPath: URL) throws -> URL {
        let fileManager = FileManager.default
        let pathExtension = modelPath.pathExtension.lowercased()
        
        // If already compiled (.mlmodelc), return as-is
        if pathExtension == "mlmodelc" {
            print("Model is already compiled.")
            return modelPath
        }
        
        // Check if it's an .mlpackage or .mlmodel that needs compilation
        guard pathExtension == "mlpackage" || pathExtension == "mlmodel" else {
            throw NSError(domain: "SHARPModelRunner", code: 10,
                         userInfo: [NSLocalizedDescriptionKey: "Unsupported model format:  \(pathExtension).Use .mlpackage, .mlmodel, or .mlmodelc"])
        }
        
        // Create a cache directory for compiled models
        let cacheDir = fileManager.temporaryDirectory.appendingPathComponent("SHARPModelCache")
        try?  fileManager.createDirectory(at: cacheDir, withIntermediateDirectories:  true)
        
        // Generate a unique name for the compiled model based on the source path
        let modelName = modelPath.deletingPathExtension().lastPathComponent
        let compiledPath = cacheDir.appendingPathComponent("\(modelName).mlmodelc")
        
        // Check if we have a cached compiled version
        if fileManager.fileExists(atPath: compiledPath.path) {
            // Verify the cached version is newer than the source
            let sourceAttrs = try fileManager.attributesOfItem(atPath:  modelPath.path)
            let cachedAttrs = try fileManager.attributesOfItem(atPath: compiledPath.path)
            
            if let sourceDate = sourceAttrs[.modificationDate] as?  Date,
               let cachedDate = cachedAttrs[.modificationDate] as? Date,
               cachedDate >= sourceDate {
                print("Using cached compiled model at \(compiledPath.path)")
                return compiledPath
            } else {
                // Source is newer, remove old cached version
                try? fileManager.removeItem(at: compiledPath)
            }
        }
        
        // Compile the model
        print("Compiling model (this may take a moment)...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let temporaryCompiledURL = try MLModel.compileModel(at: modelPath)
        
        let compileTime = CFAbsoluteTimeGetCurrent() - startTime
        print("✓ Model compiled in \(String(format: "%.1f", compileTime))s")
        
        // Move to our cache directory
        try?  fileManager.removeItem(at: compiledPath)
        try fileManager.moveItem(at: temporaryCompiledURL, to: compiledPath)
        
        print("Compiled model cached at \(compiledPath.path)")
        return compiledPath
    }
    
    /// Load and preprocess an image for model input
    func preprocessImage(at imagePath: URL) throws -> MLMultiArray {
        guard let nsImage = NSImage(contentsOf: imagePath) else {
            throw NSError(domain: "SHARPModelRunner", code: 1,
                         userInfo: [NSLocalizedDescriptionKey:  "Failed to load image from \(imagePath.path)"])
        }
        
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints:  nil) else {
            throw NSError(domain: "SHARPModelRunner", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to convert to CGImage"])
        }
        
        // Create CIImage and resize
        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()
        
        // Scale to target size
        let scaleX = CGFloat(inputWidth) / ciImage.extent.width
        let scaleY = CGFloat(inputHeight) / ciImage.extent.height
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y:  scaleY))
        
        // Render to bitmap
        guard let resizedCGImage = context.createCGImage(scaledImage, from:  CGRect(x: 0, y: 0,
                                                                                    width: inputWidth,
                                                                                    height: inputHeight)) else {
            throw NSError(domain: "SHARPModelRunner", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }
        
        // Convert to MLMultiArray (1, 3, H, W) normalized to [0, 1]
        let imageArray = try MLMultiArray(shape: [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)],
                                          dataType: .float32)
        
        let width = resizedCGImage.width
        let height = resizedCGImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgContext = CGContext(data: &pixelData,
                                        width: width,
                                        height:  height,
                                        bitsPerComponent: 8,
                                        bytesPerRow: bytesPerRow,
                                        space: colorSpace,
                                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw NSError(domain: "SHARPModelRunner", code: 4,
                         userInfo:  [NSLocalizedDescriptionKey: "Failed to create bitmap context"])
        }
        
        cgContext.draw(resizedCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Copy pixel data to MLMultiArray in CHW format
        // Use pointer access for better performance
        let ptr = imageArray.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = inputHeight * inputWidth
        
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                let r = Float(pixelData[pixelIndex]) / 255.0
                let g = Float(pixelData[pixelIndex + 1]) / 255.0
                let b = Float(pixelData[pixelIndex + 2]) / 255.0
                
                let spatialIndex = y * inputWidth + x
                ptr[0 * channelStride + spatialIndex] = r
                ptr[1 * channelStride + spatialIndex] = g
                ptr[2 * channelStride + spatialIndex] = b
            }
        }
        
        return imageArray
    }
    
    /// Run inference on the model
    func predict(image: MLMultiArray, focalLengthPx: Float) throws -> Gaussians3D {
        // Calculate disparity factor:  focal_length / image_width
        let disparityFactor = focalLengthPx / Float(inputWidth)
        
        // Create disparity factor input
        let disparityArray = try MLMultiArray(shape: [1], dataType: .float32)
        disparityArray[0] = NSNumber(value: disparityFactor)
        
        // Create feature provider
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image":  MLFeatureValue(multiArray: image),
            "disparity_factor": MLFeatureValue(multiArray:  disparityArray)
        ])
        
        // Run prediction
        let output = try model.prediction(from: inputFeatures)
        
        // Try to find outputs by checking available names
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
        
        // Helper function to find output by partial name match
        func findOutput(containing keywords: [String]) -> MLMultiArray? {
            for name in outputNames {
                let lowercaseName = name.lowercased()
                for keyword in keywords {
                    if lowercaseName.contains(keyword.lowercased()) {
                        return output.featureValue(for:  name)?.multiArrayValue
                    }
                }
            }
            return nil
        }
        
        // Try to match outputs - first try exact names, then partial matches
        let meanVectors = output.featureValue(for: "mean_vectors_3d_positions")?.multiArrayValue
            ?? findOutput(containing: ["mean", "position", "xyz"])
        
        let singularValues = output.featureValue(for: "singular_values_scales")?.multiArrayValue
            ?? findOutput(containing: ["singular", "scale"])
        
        let quaternions = output.featureValue(for: "quaternions_rotations")?.multiArrayValue
            ??  findOutput(containing:  ["quaternion", "rotation", "rot"])
        
        let colors = output.featureValue(for: "colors_rgb_linear")?.multiArrayValue
            ?? findOutput(containing: ["color", "rgb"])
        
        let opacities = output.featureValue(for: "opacities_alpha_channel")?.multiArrayValue
            ?? findOutput(containing: ["opacity", "alpha"])
        
        // If we still couldn't find outputs, try by index order
        if meanVectors == nil || singularValues == nil || quaternions == nil || colors == nil || opacities == nil {
            print("Warning: Could not match all outputs by name.Available outputs: \(outputNames)")
            
            // Try to get outputs by index if we have exactly 5
            if outputNames.count >= 5 {
                let sortedNames = outputNames.sorted()
                guard let mv = output.featureValue(for: sortedNames[0])?.multiArrayValue,
                      let sv = output.featureValue(for: sortedNames[1])?.multiArrayValue,
                      let q = output.featureValue(for: sortedNames[2])?.multiArrayValue,
                      let c = output.featureValue(for: sortedNames[3])?.multiArrayValue,
                      let o = output.featureValue(for: sortedNames[4])?.multiArrayValue else {
                    throw NSError(domain:  "SHARPModelRunner", code:  5,
                                 userInfo:  [NSLocalizedDescriptionKey: "Failed to extract model outputs. Available:  \(outputNames)"])
                }
                
                print("Using outputs by sorted order: \(sortedNames)")
                return Gaussians3D(
                    meanVectors: mv,
                    singularValues:  sv,
                    quaternions: q,
                    colors:  c,
                    opacities: o
                )
            }
            
            throw NSError(domain: "SHARPModelRunner", code: 5,
                         userInfo: [NSLocalizedDescriptionKey:  "Failed to extract model outputs.Available: \(outputNames)"])
        }
        
        return Gaussians3D(
            meanVectors: meanVectors!,
            singularValues: singularValues!,
            quaternions: quaternions!,
            colors: colors!,
            opacities:  opacities! 
        )
    }
    
    /// Save Gaussians to PLY file (matching Python save_ply format exactly)
    /// - Parameters:
    ///   - gaussians:  The Gaussians to save
    ///   - focalLengthPx:  Focal length in pixels
    ///   - imageShape: Image dimensions (height, width)
    ///   - outputPath: Output file path
    ///   - decimation: Optional decimation ratio (0.0-1.0).1.0 = keep all, 0.5 = keep 50%
    func savePLY(gaussians: Gaussians3D,
                 focalLengthPx: Float,
                 imageShape: (height: Int, width: Int),
                 to outputPath:  URL,
                 decimation: Float = 1.0) throws {
        
        let imageHeight = imageShape.height
        let imageWidth = imageShape.width
        
        // Determine which indices to keep based on decimation
        let keepIndices:  [Int]
        let originalCount = gaussians.count
        
        if decimation < 1.0 {
            keepIndices = gaussians.decimationIndices(keepRatio: decimation)
            print("Decimating:  keeping \(keepIndices.count) of \(originalCount) Gaussians (\(String(format: "%.1f", decimation * 100))%)")
        } else {
            keepIndices = Array(0..<originalCount)
        }
        
        let numGaussians = keepIndices.count
        
        var fileContent = Data()
        
        // Helper to append string
        func appendString(_ str: String) {
            fileContent.append(str.data(using: .ascii)!)
        }
        
        // Helper to append float32 in little-endian
        func appendFloat32(_ value: Float) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        
        // Helper to append int32 in little-endian
        func appendInt32(_ value: Int32) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        
        // Helper to append uint32 in little-endian
        func appendUInt32(_ value: UInt32) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        
        // Helper to append uint8
        func appendUInt8(_ value:  UInt8) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 1))
        }
        
        // ===== PLY Header =====
        appendString("ply\n")
        appendString("format binary_little_endian 1.0\n")
        
        // Vertex element
        appendString("element vertex \(numGaussians)\n")
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
        
        // Extrinsic element (16 floats for 4x4 identity matrix)
        appendString("element extrinsic 16\n")
        appendString("property float extrinsic\n")
        
        // Intrinsic element (9 floats for 3x3 matrix)
        appendString("element intrinsic 9\n")
        appendString("property float intrinsic\n")
        
        // Image size element
        appendString("element image_size 2\n")
        appendString("property uint image_size\n")
        
        // Frame element
        appendString("element frame 2\n")
        appendString("property int frame\n")
        
        // Disparity element
        appendString("element disparity 2\n")
        appendString("property float disparity\n")
        
        // Color space element
        appendString("element color_space 1\n")
        appendString("property uchar color_space\n")
        
        // Version element
        appendString("element version 3\n")
        appendString("property uchar version\n")
        
        appendString("end_header\n")
        
        // ===== Vertex Data =====
        // Compute disparity quantiles for later
        var disparities: [Float] = []
        
        // Get pointers for faster access
        let meanPtr = gaussians.meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
        let scalePtr = gaussians.singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let quatPtr = gaussians.quaternions.dataPointer.assumingMemoryBound(to: Float.self)
        let colorPtr = gaussians.colors.dataPointer.assumingMemoryBound(to:  Float.self)
        let opacityPtr = gaussians.opacities.dataPointer.assumingMemoryBound(to: Float.self)
        
        for i in keepIndices {
            // Position (x, y, z)
            let x = meanPtr[i * 3 + 0]
            let y = meanPtr[i * 3 + 1]
            let z = meanPtr[i * 3 + 2]
            appendFloat32(x)
            appendFloat32(y)
            appendFloat32(z)
            
            // Compute disparity for quantiles
            if z > 1e-6 {
                disparities.append(1.0 / z)
            }
            
            // Colors:  Convert linearRGB -> sRGB -> spherical harmonics
            // Model outputs linearRGB colors for proper alpha blending
            // We convert to sRGB for compatibility with public renderers
            let colorR = colorPtr[i * 3 + 0]
            let colorG = colorPtr[i * 3 + 1]
            let colorB = colorPtr[i * 3 + 2]
            
            let srgbR = linearRGBToSRGB(colorR)
            let srgbG = linearRGBToSRGB(colorG)
            let srgbB = linearRGBToSRGB(colorB)
            
            let sh0 = rgbToSphericalHarmonics(srgbR)
            let sh1 = rgbToSphericalHarmonics(srgbG)
            let sh2 = rgbToSphericalHarmonics(srgbB)
            
            appendFloat32(sh0)
            appendFloat32(sh1)
            appendFloat32(sh2)
            
            // Opacity:  Convert to logits using inverse sigmoid
            let opacity = opacityPtr[i]
            let opacityLogit = inverseSigmoid(opacity)
            appendFloat32(opacityLogit)
            
            // Scales:  Convert to log scale
            let scale0 = scalePtr[i * 3 + 0]
            let scale1 = scalePtr[i * 3 + 1]
            let scale2 = scalePtr[i * 3 + 2]
            
            appendFloat32(log(max(scale0, 1e-10)))
            appendFloat32(log(max(scale1, 1e-10)))
            appendFloat32(log(max(scale2, 1e-10)))
            
            // Quaternions (w, x, y, z)
            let q0 = quatPtr[i * 4 + 0]
            let q1 = quatPtr[i * 4 + 1]
            let q2 = quatPtr[i * 4 + 2]
            let q3 = quatPtr[i * 4 + 3]
            
            appendFloat32(q0)
            appendFloat32(q1)
            appendFloat32(q2)
            appendFloat32(q3)
        }
        
        // ===== Extrinsic Data (4x4 identity matrix) =====
        let identity:  [Float] = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        for val in identity {
            appendFloat32(val)
        }
        
        // ===== Intrinsic Data (3x3 matrix) =====
        let intrinsic: [Float] = [
            focalLengthPx, 0, Float(imageWidth) * 0.5,
            0, focalLengthPx, Float(imageHeight) * 0.5,
            0, 0, 1
        ]
        for val in intrinsic {
            appendFloat32(val)
        }
        
        // ===== Image Size Data =====
        appendUInt32(UInt32(imageWidth))
        appendUInt32(UInt32(imageHeight))
        
        // ===== Frame Data =====
        appendInt32(1)  // Number of frames
        appendInt32(Int32(numGaussians))  // Particles per frame
        
        // ===== Disparity Data (quantiles) =====
        disparities.sort()
        let q10Index = Int(Float(disparities.count) * 0.1)
        let q90Index = Int(Float(disparities.count) * 0.9)
        let disparity10 = disparities.isEmpty ? 0.0 : disparities[min(q10Index, disparities.count - 1)]
        let disparity90 = disparities.isEmpty ?  1.0 : disparities[min(q90Index, disparities.count - 1)]
        appendFloat32(disparity10)
        appendFloat32(disparity90)
        
        // ===== Color Space Data (sRGB = 1) =====
        appendUInt8(1)
        
        // ===== Version Data =====
        appendUInt8(1)  // Major
        appendUInt8(5)  // Minor
        appendUInt8(0)  // Patch
        
        // Write to file
        try fileContent.write(to: outputPath)
        
        print("✓ Saved PLY with \(numGaussians) Gaussians to \(outputPath.path)")
    }
}

// MARK: - Command Line Argument Parsing

struct CommandLineArgs {
    let modelPath: URL
    let imagePath: URL
    let outputPath: URL
    let focalLength: Float
    let decimation: Float
    
    static func parse() -> CommandLineArgs?  {
        let args = CommandLine.arguments
        
        var modelPath: URL? 
        var imagePath: URL?
        var outputPath: URL?
        var focalLength: Float = 1536.0
        var decimation: Float = 1.0
        
        var i = 1
        while i < args.count {
            let arg = args[i]
            
            switch arg {
            case "-m", "--model":
                i += 1
                if i < args.count {
                    modelPath = URL(fileURLWithPath: args[i])
                }
                
            case "-i", "--input":
                i += 1
                if i < args.count {
                    imagePath = URL(fileURLWithPath: args[i])
                }
                
            case "-o", "--output":
                i += 1
                if i < args.count {
                    outputPath = URL(fileURLWithPath:  args[i])
                }
                
            case "-f", "--focal-length": 
                i += 1
                if i < args.count {
                    focalLength = Float(args[i]) ?? 1536.0
                }
                
            case "-d", "--decimation": 
                i += 1
                if i < args.count {
                    if let value = Float(args[i]) {
                        // Accept both percentage (0-100) and ratio (0-1)
                        if value > 1.0 {
                            decimation = value / 100.0
                        } else {
                            decimation = value
                        }
                        decimation = max(0.01, min(1.0, decimation))
                    }
                }
                
            case "-h", "--help": 
                printUsage()
                return nil
                
            default:
                // Handle positional arguments for backward compatibility
                if modelPath == nil {
                    modelPath = URL(fileURLWithPath: arg)
                } else if imagePath == nil {
                    imagePath = URL(fileURLWithPath: arg)
                } else if outputPath == nil {
                    outputPath = URL(fileURLWithPath: arg)
                } else if focalLength == 1536.0 {
                    focalLength = Float(arg) ?? 1536.0
                }
            }
            
            i += 1
        }
        
        guard let model = modelPath, let image = imagePath, let output = outputPath else {
            printUsage()
            return nil
        }
        
        return CommandLineArgs(
            modelPath: model,
            imagePath: image,
            outputPath: output,
            focalLength: focalLength,
            decimation:  decimation
        )
    }
    
    static func printUsage() {
        let execName = CommandLine.arguments[0].components(separatedBy:  "/").last ?? "sharp_runner"
        print("""
        Usage: \(execName) [OPTIONS] <model> <input_image> <output.ply>
        
        SHARP Model Inference - Generate 3D Gaussian Splats from a single image
        
        Arguments:
          model              Path to the SHARP Core ML model (.mlpackage, .mlmodel, or .mlmodelc)
          input_image        Path to input image (PNG, JPEG, etc.)
          output.ply         Path for output PLY file
        
        Options: 
          -m, --model PATH           Path to Core ML model
          -i, --input PATH           Path to input image
          -o, --output PATH          Path for output PLY file
          -f, --focal-length FLOAT   Focal length in pixels (default: 1536)
          -d, --decimation FLOAT     Decimation ratio 0.0-1.0 or percentage 1-100 (default:  1.0 = keep all)
                                     Example: 0.5 or 50 keeps 50% of Gaussians
          -h, --help                 Show this help message
        
        Examples:
          # Basic usage
          \(execName) sharp.mlpackage photo.jpg output.ply
        
          # With focal length
          \(execName) sharp.mlpackage photo.jpg output.ply 768
        
          # With decimation (keep 50% of points)
          \(execName) -m sharp.mlpackage -i photo.jpg -o output.ply -d 0.5
        
          # With decimation as percentage
          \(execName) -m sharp.mlpackage -i photo.jpg -o output.ply -d 25
        
        The model will be automatically compiled on first use and cached for subsequent runs.
        Decimation keeps the most important Gaussians based on scale and opacity.
        """)
    }
}

// MARK:  - Main Entry Point

func main() {
    guard let args = CommandLineArgs.parse() else {
        exit(1)
    }
    
    do {
        print("Loading SHARP model from \(args.modelPath.path)...")
        let runner = try SHARPModelRunner(modelPath:  args.modelPath)
        
        print("Preprocessing image \(args.imagePath.path)...")
        let imageArray = try runner.preprocessImage(at: args.imagePath)
        
        print("Running inference...")
        let startTime = CFAbsoluteTimeGetCurrent()
        let gaussians = try runner.predict(image: imageArray, focalLengthPx: args.focalLength)
        let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("✓ Generated \(gaussians.count) Gaussians in \(String(format: "%.2f", inferenceTime))s")
        
        print("Saving PLY file...")
        try runner.savePLY(
            gaussians: gaussians,
            focalLengthPx: args.focalLength,
            imageShape: (height: 1536, width: 1536),
            to: args.outputPath,
            decimation:  args.decimation
        )
        
        print("✓ Complete!")
        
    } catch {
        print("Error: \(error.localizedDescription)")
        if let nsError = error as NSError? {
            print("Domain: \(nsError.domain), Code: \(nsError.code)")
            if let underlyingError = nsError.userInfo[NSUnderlyingErrorKey] as?  Error {
                print("Underlying error: \(underlyingError)")
            }
        }
        exit(1)
    }
}

main()
