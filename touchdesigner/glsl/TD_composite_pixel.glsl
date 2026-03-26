// TouchDesigner GLSL TOP — Pixel Shader (Pass 2: painterly composite)
// Inputs:
//   [0] original (same as structure source)
//   [1] structure pass (tangent.xy, magnitude in .z from TD_structure_pixel.glsl)
//   [2] coarse blur (large radius — maps to largest brush radius)
//   [3] mid blur
//   [4] fine blur (smallest radius)
//
// Uniforms mirror painterly_app paintify() knobs conceptually (not identical algorithm).

#version 330

uniform sampler2D sTD2DInputs[8];

// Maps from Python paintify() concepts — see touchdesigner/README.md
uniform float uThreshold;       // edge/detail threshold (try 0.15–0.55 on compressed mag)
uniform float uThresholdSoft;   // smoothstep width; larger = softer region boundaries
uniform float uCurvature;       // 0 = isotropic only, 1 = full tangent smear mix
uniform float uOpacity;         // final mix: dry/wet toward painted look
uniform float uAnisoLength;     // tangent smear length in pixels (e.g. 2–24)
uniform float uEdgeFineMix;     // 0–1 how much fine blur + original at strong edges
uniform float uUnderpaintMid;   // 0–1 extra mid-blur in flat regions (underpaint feel)

in vec3 vUV;

out vec4 fragColor;

void main() {
    vec2 uv = vUV.st;

    vec3 orig = texture(sTD2DInputs[0], uv).rgb;
    vec4 stru = texture(sTD2DInputs[1], uv);
    vec2 tangent = normalize(stru.xy + 1e-6);
    float edgeMag = stru.z;

    vec3 bCoarse = texture(sTD2DInputs[2], uv).rgb;
    vec3 bMid = texture(sTD2DInputs[3], uv).rgb;
    vec3 bFine = texture(sTD2DInputs[4], uv).rgb;

    ivec2 dim = textureSize(sTD2DInputs[0], 0);
    vec2 px = 1.0 / vec2(max(dim, ivec2(1)));
    float lenPx = max(uAnisoLength, 0.0);

    // Multi-scale blend: flat → coarse; rising edge → mid; strong edge → fine (+ orig)
    float t0 = max(uThreshold, 1e-4);
    float soft = max(uThresholdSoft, 1e-4);
    float wMid = smoothstep(t0 - soft, t0 + soft, edgeMag);
    float wFine = smoothstep(t0, t0 + 2.0 * soft, edgeMag);

    vec3 layered = mix(bCoarse, bMid, wMid);
    layered = mix(layered, bFine, wFine);
    layered = mix(layered, orig, wFine * clamp(uEdgeFineMix, 0.0, 1.0));
    float flat = 1.0 - wMid;
    layered = mix(layered, bMid, clamp(uUnderpaintMid, 0.0, 1.0) * flat);

    // Anisotropic smear along tangent (imitates flow-aligned strokes)
    vec3 smear = vec3(0.0);
    const int N = 6;
    float wsum = 0.0;
    for (int i = -N; i <= N; ++i) {
        float fi = float(i);
        float w = exp(-0.5 * fi * fi / 4.0);
        vec2 off = tangent * (fi * lenPx * px);
        smear += texture(sTD2DInputs[0], uv + off).rgb * w;
        wsum += w;
    }
    smear /= max(wsum, 1e-6);

    vec3 painted = mix(layered, smear, clamp(uCurvature, 0.0, 1.0));

    float op = clamp(uOpacity, 0.0, 1.0);
    vec3 outc = mix(orig, painted, op);

    fragColor = vec4(outc, 1.0);
}
