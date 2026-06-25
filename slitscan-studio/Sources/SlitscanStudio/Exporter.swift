// Exporter.swift — the one non-realtime operation (spec §13). Renders the
// current cut offscreen, writes PNG/TIFF/JPEG via ImageIO, and drops a recipe
// sidecar so the still can be traced back to its surface.
//
// v1 prototype renders at the working (cube) resolution. Full-source-res via
// re-decode is the deferred high-quality path (spec §8, §18.3).

import Metal
import ImageIO
import UniformTypeIdentifiers
import Foundation
import SlitscanCore

enum ExportError: Error { case renderFailed, readbackFailed, encodeFailed }

enum Exporter {
    @MainActor
    static func export(renderer: GatherRenderer, cube: Cube, params: SurfaceParams, to url: URL) throws {
        guard let tex = renderer.renderToTexture(cube: cube, params: params,
                                                 width: cube.width, height: cube.height) else {
            throw ExportError.renderFailed
        }
        guard let cg = textureToCGImage(tex) else { throw ExportError.readbackFailed }

        let type = utType(for: url)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw ExportError.encodeFailed
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw ExportError.encodeFailed }

        // Recipe sidecar: tiny, clip-independent, re-loadable.
        let recipeURL = url.deletingPathExtension().appendingPathExtension("recipe.json")
        try Session.save(params: params, clipURL: nil, to: recipeURL)
    }

    private static func utType(for url: URL) -> UTType {
        switch url.pathExtension.lowercased() {
        case "tif", "tiff": return .tiff
        case "jpg", "jpeg": return .jpeg
        default: return .png
        }
    }

    private static func textureToCGImage(_ tex: MTLTexture) -> CGImage? {
        let w = tex.width, h = tex.height
        let bytesPerRow = w * 4
        var raw = [UInt8](repeating: 0, count: bytesPerRow * h)
        let region = MTLRegionMake2D(0, 0, w, h)
        raw.withUnsafeMutableBytes { ptr in
            tex.getBytes(ptr.baseAddress!, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: info, provider: provider, decode: nil,
                       shouldInterpolate: false, intent: .defaultIntent)
    }
}
