// AppState.swift — single source of truth for the session: the cube, the live
// params (the recipe), and ingest/export orchestration. The renderer reads
// `params` + `cube` from here every frame.

import SwiftUI
import Metal
import SlitscanCore

@MainActor
final class AppState: ObservableObject {
    let device: MTLDevice
    let renderer: GatherRenderer

    @Published var params = SurfaceParams()
    @Published private(set) var cube: Cube?
    @Published var thumbnails: [NSImage] = []
    @Published var status: String = "Load the demo clip, or open a video."
    @Published var isLoading = false
    @Published private(set) var clipURL: URL?

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let renderer = GatherRenderer(device: device) else { return nil }
        self.device = device
        self.renderer = renderer
        renderer.appState = self
    }

    func loadDemo() {
        isLoading = true
        status = "Generating demo clip…"
        clipURL = nil
        let device = self.device
        Task.detached {
            let result = VideoIngest.makeDemoCube(device: device)
            await MainActor.run { self.apply(result, label: "demo clip") }
        }
    }

    func loadVideo(url: URL) {
        isLoading = true
        status = "Loading \(url.lastPathComponent)…"
        clipURL = url
        let device = self.device
        let queue = self.renderer.queue
        Task.detached {
            do {
                let result = try await VideoIngest.ingestVideo(url: url, device: device, queue: queue)
                await MainActor.run { self.apply(result, label: url.lastPathComponent) }
            } catch {
                await MainActor.run {
                    self.isLoading = false
                    self.status = "Failed to load: \(error.localizedDescription)"
                }
            }
        }
    }

    private func apply(_ result: IngestResult?, label: String) {
        isLoading = false
        guard let result else { status = "Ingest failed."; return }
        cube = result.cube
        thumbnails = result.thumbnails
        let mb = result.cube.byteSize / (1024 * 1024)
        status = "\(label) — \(result.cube.width)×\(result.cube.height), "
            + "\(result.cube.frameCount) frames, ~\(mb) MB resident"
    }

    func resetParams() {
        params = SurfaceParams()
    }

    func export(to url: URL) {
        guard let cube else { status = "Nothing to export — load a clip first."; return }
        do {
            try Exporter.export(renderer: renderer, cube: cube, params: params, to: url)
            status = "Exported \(url.lastPathComponent) (+ recipe sidecar)"
        } catch {
            status = "Export failed: \(error.localizedDescription)"
        }
    }

    func saveRecipe(to url: URL) {
        do {
            try Session.save(params: params, clipURL: clipURL, to: url)
            status = "Saved recipe \(url.lastPathComponent)"
        } catch {
            status = "Save failed: \(error.localizedDescription)"
        }
    }

    func loadRecipe(from url: URL) {
        do {
            let session = try Session.load(from: url)
            params = session.params
            status = "Loaded recipe \(url.lastPathComponent)"
            if let path = session.clipPath, cube == nil {
                let clip = URL(fileURLWithPath: path)
                if FileManager.default.fileExists(atPath: path) { loadVideo(url: clip) }
            }
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
    }
}
