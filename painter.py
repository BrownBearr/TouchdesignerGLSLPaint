from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import time

from .utils import ResizeInfo, resize_max_side, resize_to


class BrushTexture:
    def __init__(self, path: str) -> None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError("brush texture must be RGB/RGBA")

        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            rgba = np.dstack([img, alpha])
        else:
            rgba = img

        self.rgba_u8 = rgba
        self.h, self.w = rgba.shape[:2]
        if self.h < 2 or self.w < 2:
            raise ValueError("brush texture too small")

        # White = paint, black = transparent. Use luminance as coverage.
        bgr = rgba[:, :, :3].astype(np.float32)
        lum = (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]).astype(np.float32)
        self.coverage = (lum / 255.0).clip(0.0, 1.0)
        self.alpha = (rgba[:, :, 3].astype(np.float32) / 255.0).clip(0.0, 1.0)

        self._resized_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_resized(self, radius: int) -> tuple[np.ndarray, np.ndarray]:
        r = int(max(1, radius))
        if r in self._resized_cache:
            return self._resized_cache[r]

        # Map stroke radius to texture diameter.
        d = int(max(3, round(2.0 * r)))
        cov = cv2.resize(self.coverage, (d, d), interpolation=cv2.INTER_AREA)
        alp = cv2.resize(self.alpha, (d, d), interpolation=cv2.INTER_AREA)
        self._resized_cache[r] = (cov.astype(np.float32), alp.astype(np.float32))
        return self._resized_cache[r]


def _alpha_over(canvas_bgr_f32: np.ndarray, x0: int, y0: int, stamp_rgb: np.ndarray, stamp_a: np.ndarray) -> None:
    h, w = canvas_bgr_f32.shape[:2]
    sh, sw = stamp_a.shape[:2]
    if sh == 0 or sw == 0:
        return

    x1 = x0 + sw
    y1 = y0 + sh
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(w, x1)
    cy1 = min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    a = stamp_a[sy0:sy1, sx0:sx1][:, :, None]
    if float(a.max()) <= 0.0:
        return
    rgb = stamp_rgb[sy0:sy1, sx0:sx1]
    canvas = canvas_bgr_f32[cy0:cy1, cx0:cx1]
    canvas_bgr_f32[cy0:cy1, cx0:cx1] = canvas * (1.0 - a) + rgb * a


def _rotate_float_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(c, float(angle_deg), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _render_stroke_textured(
    canvas_bgr_f32: np.ndarray,
    pts: list[tuple[int, int]],
    radius: int,
    color_bgr: tuple[int, int, int],
    opacity: float,
    brush: BrushTexture,
) -> None:
    if len(pts) < 2:
        return
    cov0, alp0 = brush.get_resized(radius)
    d = cov0.shape[0]
    half = d // 2

    for (x_a, y_a), (x_b, y_b) in zip(pts[:-1], pts[1:]):
        dx = float(x_b - x_a)
        dy = float(y_b - y_a)
        if abs(dx) + abs(dy) < 1e-6:
            continue
        angle = np.degrees(np.arctan2(dy, dx))

        cov = _rotate_float_image(cov0, angle)
        alp = _rotate_float_image(alp0, angle)

        # Use alpha as the actual coverage mask and use texture luminance only to
        # modulate paint brightness (avoid multiplying coverage twice).
        #
        # This makes line/textured brushes show up instead of collapsing into
        # solid/black blobs.
        a = (alp * float(opacity)).clip(0.0, 1.0)
        if float(a.max()) <= 0.0:
            continue

        stamp_rgb = np.empty((d, d, 3), dtype=np.float32)
        # Luminance -> paint brightness multiplier
        mod = (0.15 + 0.85 * cov).astype(np.float32)
        stamp_rgb[:, :, 0] = float(color_bgr[0]) * mod
        stamp_rgb[:, :, 1] = float(color_bgr[1]) * mod
        stamp_rgb[:, :, 2] = float(color_bgr[2]) * mod

        _alpha_over(canvas_bgr_f32, x_a - half, y_a - half, stamp_rgb, a)

@dataclass(frozen=True)
class PaintParams:
    brush_radii: list[int] = None  # set in __post_init__
    threshold: float = 50.0
    max_stroke_length: int = 16
    min_stroke_length: int = 4
    curvature: float = 1.0
    opacity: float = 0.9
    grid_factor: float = 1.0
    brush_texture_path: Optional[str] = None
    fast_preview: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        if self.brush_radii is None:
            object.__setattr__(self, "brush_radii", [8, 4, 2])


def _as_int_list(values: Iterable[int]) -> list[int]:
    out = [int(v) for v in values]
    if not out:
        raise ValueError("brush_radii must be a non-empty list of ints")
    return out


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape={getattr(img, 'shape', None)}")
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        img255 = np.clip(img, 0.0, 255.0)
        return img255.astype(np.uint8)
    return img.astype(np.uint8)


def _bgr_to_lab_f32(bgr_u8: np.ndarray) -> np.ndarray:
    lab_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    return lab_u8.astype(np.float32)


def _lab_error_map(lab_ref: np.ndarray, lab_canvas: np.ndarray) -> np.ndarray:
    diff = lab_ref - lab_canvas
    return np.sqrt(np.sum(diff * diff, axis=2))


def _gaussian_blur_for_radius(bgr: np.ndarray, radius: int, sigma_scale: float = 0.5) -> np.ndarray:
    r = int(max(1, radius))
    sigma = max(0.1, float(r) * float(sigma_scale))
    k = int(max(3, (2 * int(round(sigma * 2.5)) + 1)))
    k = min(k, 61)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def _compute_intensity_f32(bgr_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _compute_gradients(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def _choose_start_in_cell(
    err: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    patch = err[y0:y1, x0:x1]
    if patch.size == 0:
        return x0, y0, 0.0
    patch_mean = float(patch.mean())
    noise = rng.random(patch.shape, dtype=np.float32) * 1e-3
    idx = int(np.argmax(patch + noise))
    py, px = divmod(idx, patch.shape[1])
    return x0 + int(px), y0 + int(py), patch_mean


def _lab_dist_1px(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    d = lab_a.astype(np.float32) - lab_b.astype(np.float32)
    return float(np.sqrt(np.sum(d * d)))


def _make_curved_stroke(
    x0: int,
    y0: int,
    radius: int,
    bgr_ref_blur: np.ndarray,
    lab_ref_blur: np.ndarray,
    canvas_bgr_f32: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gmag: np.ndarray,
    *,
    max_len: int,
    min_len: int,
    curvature: float,
) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    h, w = bgr_ref_blur.shape[:2]
    x = int(np.clip(x0, 0, w - 1))
    y = int(np.clip(y0, 0, h - 1))

    stroke_color = tuple(int(c) for c in bgr_ref_blur[y, x].tolist())
    lab_stroke = lab_ref_blur[y, x]

    pts: list[tuple[int, int]] = [(x, y)]
    last_dx, last_dy = 0.0, 0.0

    curvature = _clamp01(curvature)
    step = max(1, int(radius))

    for i in range(1, int(max_len) + 1):
        if x < 0 or y < 0 or x >= w or y >= h:
            break

        if i >= int(min_len):
            # Termination check (performance-critical): use fast BGR L1 distance like
            # the common Python references, while LAB remains the main error metric
            # for stroke placement (thresholding) and layer error maps.
            ref_bgr = bgr_ref_blur[y, x].astype(np.int16)
            can_bgr = canvas_bgr_f32[y, x].astype(np.int16)
            stroke_bgr = np.array(stroke_color, dtype=np.int16)
            d_can = int(np.sum(np.abs(ref_bgr - can_bgr)))
            d_stroke = int(np.sum(np.abs(ref_bgr - stroke_bgr)))
            if d_can <= d_stroke:
                break

        ms = float(gmag[y, x])
        if ms * step < 1e-3:
            break

        nx = float(-gy[y, x])
        ny = float(gx[y, x])

        if (last_dx * nx + last_dy * ny) < 0.0:
            nx, ny = -nx, -ny

        nx = curvature * nx + (1.0 - curvature) * last_dx
        ny = curvature * ny + (1.0 - curvature) * last_dy
        nms = float(np.hypot(nx, ny))
        if nms < 1e-6:
            break
        nx /= nms
        ny /= nms

        x = int(round(x + step * nx))
        y = int(round(y + step * ny))
        last_dx, last_dy = nx, ny

        if x < 0 or y < 0 or x >= w or y >= h:
            break
        pts.append((x, y))

    return pts, stroke_color


def _render_stroke_solid(
    canvas_bgr_f32: np.ndarray,
    pts: list[tuple[int, int]],
    radius: int,
    color_bgr: tuple[int, int, int],
    opacity: float,
    *,
    rng: Optional[np.random.Generator] = None,
    built_in_stroke_fuzz: float = 0.0,
    built_in_stroke_fuzz_blur: float = 1.0,
) -> None:
    if len(pts) < 2:
        return
    h, w = canvas_bgr_f32.shape[:2]
    thickness = int(max(1, round(2 * radius)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = int(max(radius, thickness) + 2)
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(w, max(xs) + pad + 1)
    y1 = min(h, max(ys) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    pw = x1 - x0
    ph = y1 - y0
    mask = np.zeros((ph, pw), dtype=np.float32)

    # Draw stroke into patch-space coordinates.
    for (x_a, y_a), (x_b, y_b) in zip(pts[:-1], pts[1:]):
        cv2.line(
            mask,
            (int(x_a - x0), int(y_a - y0)),
            (int(x_b - x0), int(y_b - y0)),
            1.0,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.circle(mask, (int(pts[0][0] - x0), int(pts[0][1] - y0)), radius, 1.0, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(
        mask, (int(pts[-1][0] - x0), int(pts[-1][1] - y0)), radius, 1.0, thickness=-1, lineType=cv2.LINE_AA
    )

    if built_in_stroke_fuzz > 0.0 and rng is not None:
        fuzz = _procedural_stroke_fuzz_mask(
            mask.shape,
            rng,
            strength=built_in_stroke_fuzz,
            blur_sigma=built_in_stroke_fuzz_blur,
        )
        mask = mask * fuzz

    a = (mask * float(opacity)).clip(0.0, 1.0)[:, :, None]
    if float(a.max()) <= 0.0:
        return

    patch = canvas_bgr_f32[y0:y1, x0:x1]
    stroke = np.empty_like(patch)
    stroke[:, :, 0] = float(color_bgr[0])
    stroke[:, :, 1] = float(color_bgr[1])
    stroke[:, :, 2] = float(color_bgr[2])
    canvas_bgr_f32[y0:y1, x0:x1] = patch * (1.0 - a) + stroke * a


def _procedural_stroke_fuzz_mask(
    shape: tuple[int, int],
    rng: np.random.Generator,
    strength: float,
    blur_sigma: float,
) -> np.ndarray:
    """
    Smooth random grain in [1-strength, 1] (mean ~1) to modulate stroke opacity
    without any PNG — gives a soft bristle / paper-tooth feel when strength > 0.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return np.ones(shape, dtype=np.float32)
    ph, pw = int(shape[0]), int(shape[1])
    grain = rng.random((ph, pw), dtype=np.float32)
    if blur_sigma > 0.05:
        k = int(max(3, min(31, round(float(blur_sigma) * 4) * 2 + 1)))
        if k % 2 == 0:
            k += 1
        grain = cv2.GaussianBlur(grain, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    # Blend between uniform 1.0 and grain so mean stays near 1
    mult = (1.0 - strength) + strength * grain
    return mult.astype(np.float32)


def paintify(
    bgr_uint8: np.ndarray,
    *,
    brush_radii: Iterable[int] = (8, 4, 2),
    threshold: float = 50.0,
    max_stroke_length: int = 16,
    min_stroke_length: int = 4,
    curvature: float = 1.0,
    opacity: float = 0.9,
    grid_factor: float = 1.0,
    brush_texture_path: Optional[str] = None,
    fast_preview: bool = False,
    seed: Optional[int] = None,
    preview_max_side: int = 1200,
    time_budget_s: Optional[float] = None,
    max_strokes_per_layer: Optional[int] = None,
    underpaint: bool = True,
    underpaint_mode: str = "average",  # "average" or "blur"
    force_coverage: bool = True,
    built_in_stroke_fuzz: float = 0.0,
    built_in_stroke_fuzz_blur: float = 1.0,
) -> np.ndarray:
    """
    Stroke-based painterly rendering (Hertzmann 1998), OpenCV/NumPy implementation.

    Inputs/outputs are BGR uint8 (OpenCV default).
    """
    src_full = _ensure_bgr_uint8(bgr_uint8)
    src, resize_info = resize_max_side(src_full, preview_max_side)
    h, w = src.shape[:2]

    radii = sorted(_as_int_list(brush_radii), reverse=True)
    if fast_preview and radii:
        radii = [radii[0]]

    rng = np.random.default_rng(seed)

    # Underpainting ensures the full canvas is filled immediately.
    # Use "average" by default so strokes still get placed (blur-underpaint can reduce initial error too much).
    if underpaint and radii:
        mode = (underpaint_mode or "average").strip().lower()
        if mode == "blur":
            base = _gaussian_blur_for_radius(src, int(max(radii)), sigma_scale=0.5)
            canvas = base.astype(np.float32)
        else:
            ave = src.reshape(-1, 3).mean(axis=0)
            canvas = np.full((h, w, 3), ave.astype(np.float32), dtype=np.float32)
    else:
        canvas = np.full((h, w, 3), 255.0, dtype=np.float32)

    brush: Optional[BrushTexture] = None
    if brush_texture_path:
        brush = BrushTexture(brush_texture_path)

    t0 = time.time()

    for r in radii:
        r = int(max(1, r))

        ref_blur = _gaussian_blur_for_radius(src, r, sigma_scale=0.5)
        lab_ref = _bgr_to_lab_f32(ref_blur)

        gray = _compute_intensity_f32(ref_blur)
        gx, gy, gmag = _compute_gradients(gray)

        lab_canvas = _bgr_to_lab_f32(canvas.astype(np.uint8))
        err = _lab_error_map(lab_ref, lab_canvas)

        grid = int(max(1, round(float(r) * float(grid_factor))))
        cell_coords: list[tuple[int, int]] = []
        for y0 in range(0, h, grid):
            for x0 in range(0, w, grid):
                cell_coords.append((x0, y0))
        rng.shuffle(cell_coords)

        strokes_painted = 0
        is_first_layer = r == radii[0]
        for x0, y0 in cell_coords:
            if time_budget_s is not None and (time.time() - t0) >= float(time_budget_s):
                break
            if max_strokes_per_layer is not None and strokes_painted >= int(max_strokes_per_layer):
                break

            x1 = min(w, x0 + grid)
            y1 = min(h, y0 + grid)
            sx, sy, mean_err = _choose_start_in_cell(err, x0, y0, x1, y1, rng)
            if mean_err <= float(threshold):
                # Bright/flat regions can have low error early, which may leave
                # noticeable unpainted gaps under a time/stroke budget. Optionally
                # force a base layer of coverage.
                if not (force_coverage and is_first_layer):
                    continue

            pts, color = _make_curved_stroke(
                sx,
                sy,
                r,
                ref_blur,
                lab_ref,
                canvas,
                gx,
                gy,
                gmag,
                max_len=max_stroke_length,
                min_len=min_stroke_length,
                curvature=curvature,
            )

            if brush is not None:
                _render_stroke_textured(canvas, pts, r, color, opacity, brush)
            else:
                _render_stroke_solid(
                    canvas,
                    pts,
                    r,
                    color,
                    opacity,
                    rng=rng,
                    built_in_stroke_fuzz=float(built_in_stroke_fuzz),
                    built_in_stroke_fuzz_blur=float(built_in_stroke_fuzz_blur),
                )
            strokes_painted += 1

    out_small = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    if resize_info.scale != 1.0:
        out_full = resize_to(out_small, resize_info.orig_size)
        return out_full
    return out_small


def _demo_main() -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Painterly stroke-based rendering demo")
    parser.add_argument("image", type=str, help="Path to an input image")
    parser.add_argument("--out", type=str, default="out_painterly.png")
    parser.add_argument("--radii", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--max_len", type=int, default=16)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--opacity", type=float, default=0.9)
    parser.add_argument("--grid_factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inp = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if inp is None:
        raise FileNotFoundError(args.image)

    t0 = time.time()
    out = paintify(
        inp,
        brush_radii=args.radii,
        threshold=args.threshold,
        max_stroke_length=args.max_len,
        min_stroke_length=args.min_len,
        curvature=args.curvature,
        opacity=args.opacity,
        grid_factor=args.grid_factor,
        seed=args.seed,
    )
    dt = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Saved {out_path} in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo_main())
