// GatherUniforms.swift — Swift mirror of the MSL struct (same field order &
// all-float layout, so memory layout matches without a bridging header).

import simd
import SlitscanCore

struct GatherUniforms {
    var profile: Float = 0
    var axis: Float = 0
    var vanguard: Float = 0
    var spread: Float = 0.5
    var bufferOrigin: Float = 1
    var sliceWidth: Float = 1
    var fillMode: Float = 3      // hold
    var interpolate: Float = 0
    var frameCount: Float = 1
    var texWidth: Float = 1
    var texHeight: Float = 1
    var _pad: Float = 0

    init() {}

    init(params: SurfaceParams, frameCount: Int, width: Int, height: Int) {
        profile = params.profile.shaderCode
        axis = params.axis.shaderCode
        vanguard = Float(params.vanguard)
        spread = Float(params.spread)
        bufferOrigin = Float(params.bufferOrigin)
        sliceWidth = Float(max(params.sliceWidth, 1))
        fillMode = params.fill.shaderCode
        interpolate = params.interpolate ? 1 : 0
        self.frameCount = Float(max(frameCount, 1))
        texWidth = Float(max(width, 1))
        texHeight = Float(max(height, 1))
    }
}
