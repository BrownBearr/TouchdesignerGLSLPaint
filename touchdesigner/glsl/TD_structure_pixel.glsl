// TouchDesigner GLSL TOP — Pixel Shader (Pass 1: structure)
// Inputs: [0] source RGB(A) video/image
// Output: RGBA — tangent.xy (unit vector along stroke direction), Z = gradient magnitude, W = 1
// Tip: set this TOP to 16-bit float (RGBA16F) so magnitude in Z is not clipped.

#version 330

uniform sampler2D sTD2DInputs[8];

uniform float uMagNormalize;
// ~0.02–0.15 typical; higher = edge magnitudes saturate faster in .z for 8-bit buffers

in vec3 vUV;

out vec4 fragColor;

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    ivec2 dim = textureSize(sTD2DInputs[0], 0);
    vec2 px = 1.0 / vec2(max(dim, ivec2(1)));

    vec2 uv = vUV.st;

    float s00 = luminance(texture(sTD2DInputs[0], uv + vec2(-px.x, -px.y)).rgb);
    float s01 = luminance(texture(sTD2DInputs[0], uv + vec2(0.0, -px.y)).rgb);
    float s02 = luminance(texture(sTD2DInputs[0], uv + vec2(px.x, -px.y)).rgb);
    float s10 = luminance(texture(sTD2DInputs[0], uv + vec2(-px.x, 0.0)).rgb);
    float s12 = luminance(texture(sTD2DInputs[0], uv + vec2(px.x, 0.0)).rgb);
    float s20 = luminance(texture(sTD2DInputs[0], uv + vec2(-px.x, px.y)).rgb);
    float s21 = luminance(texture(sTD2DInputs[0], uv + vec2(0.0, px.y)).rgb);
    float s22 = luminance(texture(sTD2DInputs[0], uv + vec2(px.x, px.y)).rgb);

    // Sobel
    float gx = -s00 + s02 - 2.0 * s10 + 2.0 * s12 - s20 + s22;
    float gy = -s00 - 2.0 * s01 - s02 + s20 + 2.0 * s21 + s22;

    float mag = length(vec2(gx, gy));
    vec2 g = vec2(gx, gy);
    float len = length(g);
    vec2 tangent = len > 1e-6 ? vec2(-g.y, g.x) / len : vec2(1.0, 0.0);

    float mn = max(uMagNormalize, 1e-6);
    float magOut = 1.0 - exp(-mag * mn);

    fragColor = vec4(tangent.xy, magOut, 1.0);
}
