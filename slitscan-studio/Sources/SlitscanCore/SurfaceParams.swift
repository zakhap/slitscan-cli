// SurfaceParams.swift — the "recipe": the full parameter set defining one cut.
// Tiny, Codable, clip-independent (spec §14).

import Foundation

public enum Axis: String, Codable, CaseIterable, Sendable {
    case x
    case y
    public var shaderCode: Float { self == .x ? 0 : 1 }
}

public enum FillMode: String, Codable, CaseIterable, Sendable {
    case black
    case white
    case transparent
    case hold
    case wrap

    /// Must match Gather.metal's fill switch.
    public var shaderCode: Float {
        switch self {
        case .black: return 0
        case .white: return 1
        case .transparent: return 2
        case .hold: return 3
        case .wrap: return 4
        }
    }
}

/// The editable state of a single still. `buffer_origin` and `spread` are the
/// two separate time controls (spec §11); both 0..1.
public struct SurfaceParams: Codable, Equatable, Sendable {
    public var profile: Profile = .ramp
    public var axis: Axis = .x

    /// 0..1 position of the delay-0 "now" locus.
    public var vanguard: Double = 0.0
    /// 0..1 — how far back in time the surface rakes. 0 = a single instant.
    public var spread: Double = 0.5
    /// 0..1 — where the sampled window sits in the clip (the scrub).
    public var bufferOrigin: Double = 1.0

    /// Band size in pixels (the grain / venetian-blind control).
    public var sliceWidth: Int = 1
    public var fill: FillMode = .hold
    public var interpolate: Bool = false

    public init() {}
}
