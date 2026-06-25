// SelfTest.swift — headless end-to-end check (no window required).
// Builds the synthetic demo cube, runs the REAL gather shader offscreen for a
// couple of cuts, exports PNGs, and verifies the result is an actual slit-scan
// (source time varies across the image — not a flat frame copy).
//
// Run: `swift run SlitscanStudio --selftest`

import Metal
import Foundation
import SlitscanCore

enum SelfTest {
    @MainActor
    static func run() {
        func fail(_ m: String) -> Never { FileHandle.standardError.write(Data("SELFTEST FAIL: \(m)\n".utf8)); exit(1) }

        guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
        guard let renderer = GatherRenderer(device: device) else { fail("renderer init failed (shader compile?)") }
        guard let result = VideoIngest.makeDemoCube(device: device) else { fail("demo cube failed") }
        let cube = result.cube
        print("cube: \(cube.width)×\(cube.height), \(cube.frameCount) frames, ~\(cube.byteSize/(1024*1024)) MB")

        var cuts: [(String, SurfaceParams)] = []
        var p1 = SurfaceParams(); p1.profile = .ramp; p1.vanguard = 0; p1.spread = 1.0; p1.bufferOrigin = 1.0
        var p2 = SurfaceParams(); p2.profile = .tent; p2.vanguard = 0.5; p2.spread = 1.0; p2.bufferOrigin = 1.0
        cuts = [("ramp", p1), ("tent", p2)]

        for (name, params) in cuts {
            guard let tex = renderer.renderToTexture(cube: cube, params: params,
                                                     width: cube.width, height: cube.height) else {
                fail("render \(name) failed")
            }
            // Read back the middle row's blue channel (time is encoded in blue in
            // the demo cube). A real slit-scan makes blue vary across columns.
            let w = tex.width, h = tex.height
            let bytesPerRow = w * 4
            var raw = [UInt8](repeating: 0, count: bytesPerRow * h)
            raw.withUnsafeMutableBytes { ptr in
                tex.getBytes(ptr.baseAddress!, bytesPerRow: bytesPerRow,
                             from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            }
            let rowBase = (h / 2) * bytesPerRow
            var minB = 255, maxB = 0
            for x in 0..<w {
                let b = Int(raw[rowBase + x * 4])   // BGRA → blue is byte 0
                minB = min(minB, b); maxB = max(maxB, b)
            }
            let spread = maxB - minB
            let out = URL(fileURLWithPath: "/tmp/slitscan_selftest_\(name).png")
            do { try Exporter.export(renderer: renderer, cube: cube, params: params, to: out) }
            catch { fail("export \(name): \(error)") }
            print("\(name): blue range across row = \(spread) (>0 ⇒ slit-scan) → \(out.path)")
            if spread <= 0 { fail("\(name) produced a flat frame, not a time-varying cut") }
        }
        print("SELFTEST PASS")
    }
}
