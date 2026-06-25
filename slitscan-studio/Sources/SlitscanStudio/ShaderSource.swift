// ShaderSource.swift — the realtime gather, as MSL compiled at runtime.
//
// Embedded as a string (rather than a bundled .metal) so the SPM executable has
// no resource-bundle dependency. The profile arithmetic mirrors
// SlitscanCore/Profiles.swift exactly; the sign convention is pinned to the
// CLI's `src = origin - spread*delay` (spec §3, §7). Keep all three in lockstep.

let gatherShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct GatherUniforms {
    float profile;       // 0 ramp, 1 reverse, 2 tent
    float axis;          // 0 x, 1 y
    float vanguard;
    float spread;
    float bufferOrigin;
    float sliceWidth;
    float fillMode;      // 0 black,1 white,2 transparent,3 hold,4 wrap
    float interpolate;
    float frameCount;
    float texWidth;
    float texHeight;
    float pad;
};

struct VOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VOut fullscreen_vs(uint vid [[vertex_id]]) {
    float2 p[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 q = p[vid];
    VOut o;
    o.pos = float4(q, 0.0, 1.0);
    o.uv = float2((q.x + 1.0) * 0.5, 1.0 - (q.y + 1.0) * 0.5);
    return o;
}

static inline float delay01(float profile, float t, float vanguard) {
    if (profile < 0.5) {
        return (vanguard <= 0.5) ? t : (1.0 - t);
    } else if (profile < 1.5) {
        return (vanguard >= 0.5) ? (1.0 - t) : t;
    } else {
        float denom = max(max(vanguard, 1.0 - vanguard), 1e-9);
        return clamp(fabs(t - vanguard) / denom, 0.0, 1.0);
    }
}

static inline float4 fillColor(int mode) {
    if (mode == 1) return float4(1.0, 1.0, 1.0, 1.0);
    if (mode == 2) return float4(0.0, 0.0, 0.0, 0.0);
    return float4(0.0, 0.0, 0.0, 1.0);
}

static inline float4 sampleSlice(texture2d_array<float> cube, sampler s,
                                 float2 uv, int slice, int fc, int mode,
                                 thread bool &oob) {
    oob = false;
    if (slice < 0 || slice > fc - 1) {
        if (mode == 3) {
            slice = clamp(slice, 0, fc - 1);
        } else if (mode == 4) {
            slice = ((slice % fc) + fc) % fc;
        } else {
            oob = true;
            return float4(0.0);
        }
    }
    return cube.sample(s, uv, uint(slice));
}

fragment float4 gather_fs(VOut in [[stage_in]],
                          texture2d_array<float> cube [[texture(0)]],
                          constant GatherUniforms& u [[buffer(0)]],
                          sampler s [[sampler(0)]]) {
    float W = max(u.texWidth, 1.0);
    float H = max(u.texHeight, 1.0);
    float extent = (u.axis < 0.5) ? W : H;
    float axisPos = (u.axis < 0.5) ? in.uv.x : in.uv.y;

    float sliceW = max(u.sliceWidth, 1.0);
    float nBands = max(ceil(extent / sliceW), 1.0);
    float aPx = axisPos * (extent - 1.0);
    float bandIdx = floor(aPx / sliceW);
    float t = (nBands > 1.0) ? (bandIdx / (nBands - 1.0)) : 0.0;

    float d = delay01(u.profile, t, u.vanguard);

    float fc = max(u.frameCount, 1.0);
    float originFrame = u.bufferOrigin * (fc - 1.0);
    float delayFrames = d * u.spread * (fc - 1.0);
    float src = originFrame - delayFrames;       // pinned: minus

    int fci = int(fc);
    int mode = int(u.fillMode);

    if (u.interpolate > 0.5) {
        int f0 = int(floor(src));
        int f1 = f0 + 1;
        float a = src - float(f0);
        bool o0, o1;
        float4 c0 = sampleSlice(cube, s, in.uv, f0, fci, mode, o0);
        float4 c1 = sampleSlice(cube, s, in.uv, f1, fci, mode, o1);
        float4 fc4 = fillColor(mode);
        if (o0) c0 = fc4;
        if (o1) c1 = fc4;
        return mix(c0, c1, a);
    } else {
        int f0 = int(round(src));
        bool o0;
        float4 c0 = sampleSlice(cube, s, in.uv, f0, fci, mode, o0);
        return o0 ? fillColor(mode) : c0;
    }
}
"""
