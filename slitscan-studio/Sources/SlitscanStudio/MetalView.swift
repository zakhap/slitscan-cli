// MetalView.swift — the live preview canvas. Wraps an MTKView driven by the
// shared GatherRenderer, and adds direct manipulation: drag = vanguard (x) +
// spread (y), so the image is itself a control surface (spec §12).

import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    @ObservedObject var appState: AppState

    func makeCoordinator() -> GatherRenderer { appState.renderer }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: appState.device)
        view.colorPixelFormat = .bgra8Unorm
        view.delegate = context.coordinator
        view.preferredFramesPerSecond = 60
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0.07, green: 0.07, blue: 0.08, alpha: 1)

        let pan = NSPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(GatherRenderer.handlePan(_:)))
        view.addGestureRecognizer(pan)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
