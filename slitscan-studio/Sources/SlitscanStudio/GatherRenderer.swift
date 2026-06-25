// GatherRenderer.swift — drives the gather shader into an MTKView every frame,
// and offscreen for export. Reads live params + cube from AppState.

import Metal
import MetalKit
import AppKit
import SlitscanCore

@MainActor
final class GatherRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState

    weak var appState: AppState?

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue

        guard let library = try? device.makeLibrary(source: gatherShaderSource, options: nil),
              let vfn = library.makeFunction(name: "fullscreen_vs"),
              let ffn = library.makeFunction(name: "gather_fs")
        else { return nil }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }
        self.pipeline = pipeline

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear
        sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        guard let sampler = device.makeSamplerState(descriptor: sd) else { return nil }
        self.sampler = sampler

        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    /// Direct manipulation on the canvas: x → vanguard, y → spread.
    @objc func handlePan(_ g: NSPanGestureRecognizer) {
        guard let view = g.view, let appState else { return }
        let p = g.location(in: view)
        let w = max(view.bounds.width, 1)
        let h = max(view.bounds.height, 1)
        var params = appState.params
        params.vanguard = Double(min(max(p.x / w, 0), 1))
        params.spread = Double(min(max(1 - p.y / h, 0), 1))   // drag up = wider rake
        appState.params = params
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        if let cube = appState?.cube, let params = appState?.params {
            var u = GatherUniforms(params: params, frameCount: cube.frameCount,
                                   width: cube.width, height: cube.height)
            enc.setRenderPipelineState(pipeline)
            enc.setFragmentTexture(cube.texture, index: 0)
            enc.setFragmentSamplerState(sampler, index: 0)
            enc.setFragmentBytes(&u, length: MemoryLayout<GatherUniforms>.stride, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }

    /// Render the current cut to an offscreen BGRA texture for export.
    func renderToTexture(cube: Cube, params: SurfaceParams, width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .shared
        guard let target = device.makeTexture(descriptor: desc) else { return nil }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rpd.colorAttachments[0].storeAction = .store

        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return nil }

        // Export keeps the cut's surface; only resolution differs (texWidth/Height
        // here describe the OUTPUT grid, which the shader normalizes against).
        var u = GatherUniforms(params: params, frameCount: cube.frameCount,
                               width: width, height: height)
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentTexture(cube.texture, index: 0)
        enc.setFragmentSamplerState(sampler, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<GatherUniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return target
    }
}
