// Cube.swift — the X-Y-T volume held resident on the GPU as a texture2d_array
// (one slice per frame). The whole realtime story depends on this staying
// resident so re-cuts are a uniform update, not a recompute (spec §6).

import Metal
import Foundation

final class Cube {
    let texture: MTLTexture
    let width: Int
    let height: Int
    let frameCount: Int
    let fps: Double

    init(texture: MTLTexture, width: Int, height: Int, frameCount: Int, fps: Double) {
        self.texture = texture
        self.width = width
        self.height = height
        self.frameCount = frameCount
        self.fps = fps
    }

    /// Approximate resident size in bytes (RGBA8).
    var byteSize: Int { width * height * 4 * frameCount }

    /// Allocate an empty cube texture sized for `frameCount` slices.
    static func makeTexture(device: MTLDevice, width: Int, height: Int, frameCount: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .bgra8Unorm
        desc.width = width
        desc.height = height
        desc.arrayLength = max(frameCount, 1)
        desc.storageMode = .shared          // Apple-silicon unified memory
        desc.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }
}
