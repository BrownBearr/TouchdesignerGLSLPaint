// TouchDesigner GLSL TOP — Pixel Shader (Gooey Halftone / Broken Cell Wall)
//
// Basis: neighboring-cell sampling + "cell wall" break + gooey dots concept from
// Maxime Heckel, "Shades of Halftone" (2026):
// https://blog.maximeheckel.com/posts/shades-of-halftone/
//
// Input:
//   [0] source RGB(A) video/image
//
// Output:
//   RGB = color sampled from source (at dot's cell center)
//   A   = 1
//
// Visual behavior:
// - Image is partitioned into a grid of cells (uCellSizePx).
// - Each cell has a dot centered in the cell, colored from the source at that center.
// - Dot radius grows in darker regions (based on luma).
// - To avoid dots clipping to the cell boundary, each pixel considers dots from its
//   neighboring cells (3x3 by default; adjustable via uSearchRadius).
// - "Gooeyness" merges overlapping dots by combining soft coverage fields.
//
// Notes:
// - This is a stylized realtime shader, not a port of the Hertzmann stroke loop.
// - For better anti-aliasing, keep the GLSL TOP output in 16-bit float if available.

#version 330

uniform sampler2D sTD2DInputs[8];

// Grid / pattern controls
uniform vec2  uCellSizePx;     // e.g. (10,10) .. (40,40)
uniform int   uSearchRadius;   // 1 => 3x3 neighbors, 2 => 5x5 (expensive)

// Dot size controls (luma-driven)
uniform float uBaseRadius;     // base radius in pixels (e.g. 0.10 * min(cell))
uniform float uRadiusByDark;   // extra radius added for dark regions (pixels)
uniform float uGamma;          // luma shaping (e.g. 1.0..2.2). Higher => more emphasis on darks.

// Coverage / look controls
uniform float uEdgeSoftness;   // AA softness in pixels (e.g. 1.0)
uniform float uGooeyness;      // 0..1. 0=max/winner, 1=gooey accumulation
uniform vec3  uBgColor;        // background color behind dots
uniform float uInkStrength;    // 0..1 how strongly dots replace background (typically 1)

in vec3 vUV;
out vec4 fragColor;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float saturate(float x) { return clamp(x, 0.0, 1.0); }

// Smooth dot coverage from distance in pixels.
float dotMask(float distPx, float radiusPx, float edgeSoftPx) {
    // distPx < radiusPx => inside dot
    float w = max(edgeSoftPx, 1e-4);
    return 1.0 - smoothstep(radiusPx - w, radiusPx + w, distPx);
}

void main() {
    ivec2 dim = textureSize(sTD2DInputs[0], 0);
    vec2 res = vec2(max(dim, ivec2(1)));

    vec2 uv = vUV.st;
    vec2 p = uv * res; // pixel coords

    // Guard against degenerate cell sizes
    vec2 cellPx = max(uCellSizePx, vec2(1.0));

    // Identify which cell this pixel is in.
    vec2 baseCell = floor(p / cellPx);

    // Accumulate dot coverage and color.
    float best = 0.0;
    vec3 bestCol = texture(sTD2DInputs[0], uv).rgb;

    float acc = 0.0;
    vec3 accCol = vec3(0.0);

    int r = max(uSearchRadius, 0);

    // Neighbor search: breaks the "cell wall" so dots can spill across.
    for (int dy = -4; dy <= 4; ++dy) {
        if (dy < -r || dy > r) continue;
        for (int dx = -4; dx <= 4; ++dx) {
            if (dx < -r || dx > r) continue;

            vec2 cell = baseCell + vec2(float(dx), float(dy));
            vec2 centerPx = (cell + 0.5) * cellPx;

            vec2 centerUv = centerPx / res;
            vec3 src = texture(sTD2DInputs[0], centerUv).rgb;

            float y = saturate(luma(src));
            // Darker => bigger dots. Use gamma to control response.
            float dark = pow(1.0 - y, max(uGamma, 1e-3));

            float radiusPx = max(0.0, uBaseRadius + uRadiusByDark * dark);

            float d = length(p - centerPx);
            float m = dotMask(d, radiusPx, uEdgeSoftness);

            // Winner-take-all (crisp) path
            if (m > best) {
                best = m;
                bestCol = src;
            }

            // Gooey accumulation path
            acc += m;
            accCol += src * m;
        }
    }

    // Blend between winner and gooey accumulation.
    float g = saturate(uGooeyness);
    vec3 gooCol = acc > 1e-6 ? (accCol / acc) : bestCol;
    float gooMask = saturate(acc); // can exceed 1, clamp gives thicker merges

    float mask = mix(best, gooMask, g);
    vec3 col = mix(bestCol, gooCol, g);

    vec3 outRgb = mix(uBgColor, col, saturate(uInkStrength) * mask);
    fragColor = vec4(outRgb, 1.0);
}
