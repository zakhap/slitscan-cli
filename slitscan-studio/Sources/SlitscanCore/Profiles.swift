// Profiles.swift — the surface math, ported faithfully from the CLI's NumPy.
//
// These are the v1 profiles: all 1D (delay is a function of position along one
// axis). `diagonal` and `radial` are deferred to v2 because the CLI has no
// reference implementation to gate them against (see spec §3, §10).
//
// This is the canonical Swift definition. The Metal shader (Gather.metal)
// transliterates the same arithmetic; the vendored NumPy reference
// (reference/generate_vectors.py) emits test vectors this file must reproduce.
// Keep all three in lockstep.

import Foundation

public enum Profile: String, Codable, CaseIterable, Sendable {
    case ramp
    case reverse
    case tent

    /// Integer code passed to the shader uniform. Must match Gather.metal.
    public var shaderCode: Float {
        switch self {
        case .ramp: return 0
        case .reverse: return 1
        case .tent: return 2
        }
    }
}

/// Normalized delay in [0, 1] for a single position `t` (also normalized [0,1])
/// along the active axis. Multiply by the effective max-delay (frames) to get
/// the real delay. `vanguard` is the 0..1 position of the "now" locus.
///
/// Faithful to the CLI:
///   - ramp:    `vanguard <= 0.5` ? t : 1-t        (profiles/ramp.py)
///   - reverse: `vanguard >= 0.5` ? 1-t : t        (profiles/reverse.py)
///   - tent:    clip(|t-vanguard| / max(max(vanguard,1-vanguard), 1e-9), 0, 1)
public func delay01(_ profile: Profile, t: Double, vanguard: Double) -> Double {
    switch profile {
    case .ramp:
        return vanguard <= 0.5 ? t : 1.0 - t
    case .reverse:
        return vanguard >= 0.5 ? 1.0 - t : t
    case .tent:
        let denom = max(max(vanguard, 1.0 - vanguard), 1e-9)
        return min(max(abs(t - vanguard) / denom, 0.0), 1.0)
    }
}

/// Per-band normalized delays for `n` bands — mirrors the CLI's
/// `delay_map(x_coords, ...)` where `t = x_coords / max(n-1, 1)`.
public func delays(_ profile: Profile, n: Int, vanguard: Double) -> [Double] {
    let denom = Double(max(n - 1, 1))
    return (0..<n).map { delay01(profile, t: Double($0) / denom, vanguard: vanguard) }
}
