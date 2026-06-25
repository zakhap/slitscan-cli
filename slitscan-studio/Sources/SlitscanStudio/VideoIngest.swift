// VideoIngest.swift — fill the cube. Two sources:
//   1. makeDemoCube — a synthetic clip so the app runs with no external file.
//   2. ingestVideo  — decode a real clip (AVFoundation) and scale each frame
//      into a cube slice on the GPU (MPS), at a memory-budgeted working res.

import Metal
import MetalKit
import MetalPerformanceShaders
import AVFoundation
import CoreVideo
import AppKit

enum IngestError: Error { case noVideoTrack, textureAllocFailed, readerFailed }

struct IngestResult {
    let cube: Cube
    let thumbnails: [NSImage]
}

enum VideoIngest {

    // Memory budget knobs (spec §6). Defaults sized to stay comfortable on 16GB.
    static let defaultMaxLongSide = 720
    static let defaultMaxFrames = 360

    // MARK: Synthetic demo cube

    /// A moving bar + per-frame color gradient, so a slit-scan cut is obviously
    /// visible (each column reads a different moment → the bar smears diagonally).
    static func makeDemoCube(device: MTLDevice,
                             width: Int = 640, height: Int = 360,
                             frames: Int = 180, fps: Double = 30) -> IngestResult? {
        guard let texture = Cube.makeTexture(device: device, width: width, height: height, frameCount: frames) else {
            return nil
        }
        let bytesPerRow = width * 4
        var buf = [UInt8](repeating: 0, count: bytesPerRow * height)
        var thumbs: [NSImage] = []
        let thumbEvery = max(frames / 16, 1)

        for f in 0..<frames {
            let tNorm = frames > 1 ? Double(f) / Double(frames - 1) : 0
            let barX = Int(tNorm * Double(width - 1))
            for y in 0..<height {
                let g = UInt8(Double(y) / Double(max(height - 1, 1)) * 255)
                let b = UInt8(tNorm * 255)            // time encoded in blue
                let rowBase = y * bytesPerRow
                for x in 0..<width {
                    let r = UInt8(Double(x) / Double(max(width - 1, 1)) * 255)
                    let i = rowBase + x * 4
                    if abs(x - barX) <= 3 {           // bright moving bar (BGRA)
                        buf[i] = 255; buf[i+1] = 255; buf[i+2] = 255; buf[i+3] = 255
                    } else {
                        buf[i] = b; buf[i+1] = g; buf[i+2] = r; buf[i+3] = 255
                    }
                }
            }
            let region = MTLRegionMake2D(0, 0, width, height)
            buf.withUnsafeBytes { raw in
                texture.replace(region: region, mipmapLevel: 0, slice: f,
                                withBytes: raw.baseAddress!, bytesPerRow: bytesPerRow, bytesPerImage: 0)
            }
            if f % thumbEvery == 0, let img = bgraToImage(buf, width: width, height: height) {
                thumbs.append(img)
            }
        }
        let cube = Cube(texture: texture, width: width, height: height, frameCount: frames, fps: fps)
        return IngestResult(cube: cube, thumbnails: thumbs)
    }

    // MARK: Real video

    static func ingestVideo(url: URL, device: MTLDevice, queue: MTLCommandQueue,
                            maxLongSide: Int = defaultMaxLongSide,
                            maxFrames: Int = defaultMaxFrames) async throws -> IngestResult {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw IngestError.noVideoTrack }

        let natSize = try await track.load(.naturalSize)
        let nominalFPS = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration)
        let fps = nominalFPS > 0 ? Double(nominalFPS) : 30
        let durationSec = CMTimeGetSeconds(duration)

        let srcW = max(Int(natSize.width.rounded()), 2)
        let srcH = max(Int(natSize.height.rounded()), 2)
        let (W, H) = workingSize(srcW: srcW, srcH: srcH, maxLongSide: maxLongSide)

        let estFrames = max(1, min(maxFrames, Int((durationSec * fps).rounded())))
        guard let texture = Cube.makeTexture(device: device, width: W, height: H, frameCount: estFrames) else {
            throw IngestError.textureAllocFailed
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw IngestError.readerFailed }

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        let scaler = MPSImageBilinearScale(device: device)

        var filled = 0
        while filled < estFrames, let sample = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pw = CVPixelBufferGetWidth(pixelBuffer)
            let ph = CVPixelBufferGetHeight(pixelBuffer)
            var cvTex: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache!, pixelBuffer, nil,
                .bgra8Unorm, pw, ph, 0, &cvTex)
            guard let cvTex, let srcTex = CVMetalTextureGetTexture(cvTex) else { continue }
            guard let sliceView = texture.makeTextureView(
                pixelFormat: .bgra8Unorm, textureType: .type2D,
                levels: 0..<1, slices: filled..<(filled + 1)) else { continue }

            guard let cb = queue.makeCommandBuffer() else { continue }
            var transform = MPSScaleTransform(
                scaleX: Double(W) / Double(pw), scaleY: Double(H) / Double(ph),
                translateX: 0, translateY: 0)
            withUnsafePointer(to: &transform) { scaler.scaleTransform = $0 }
            scaler.encode(commandBuffer: cb, sourceTexture: srcTex, destinationTexture: sliceView)
            cb.commit()
            cb.waitUntilCompleted()
            filled += 1
        }

        let actualFrames = max(filled, 1)
        let cube = Cube(texture: texture, width: W, height: H, frameCount: actualFrames, fps: fps)
        let thumbnails = try? await makeThumbnails(asset: asset, durationSec: durationSec)
        return IngestResult(cube: cube, thumbnails: thumbnails ?? [])
    }

    // MARK: Helpers

    static func workingSize(srcW: Int, srcH: Int, maxLongSide: Int) -> (Int, Int) {
        let longSide = max(srcW, srcH)
        let scale = longSide > maxLongSide ? Double(maxLongSide) / Double(longSide) : 1.0
        func even(_ v: Double) -> Int { var i = Int(v.rounded()); i -= i % 2; return max(2, i) }
        return (even(Double(srcW) * scale), even(Double(srcH) * scale))
    }

    private static func makeThumbnails(asset: AVAsset, durationSec: Double, count: Int = 16) async throws -> [NSImage] {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 0, height: 64)
        var images: [NSImage] = []
        for k in 0..<count {
            let frac = count > 1 ? Double(k) / Double(count - 1) : 0
            let time = CMTime(seconds: durationSec * frac, preferredTimescale: 600)
            if let cg = try? await gen.image(at: time).image {
                images.append(NSImage(cgImage: cg, size: .zero))
            }
        }
        return images
    }

    static func bgraToImage(_ bytes: [UInt8], width: Int, height: Int) -> NSImage? {
        let bytesPerRow = width * 4
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let cg = CGImage(width: width, height: height, bitsPerComponent: 8,
                               bitsPerPixel: 32, bytesPerRow: bytesPerRow,
                               space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info,
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: width, height: height))
    }
}
