from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr_uint8(img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_uint8_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got {img_bgr.shape}")
    rgb = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def load_image_bgr(path: str | Path) -> np.ndarray:
    p = str(path)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def save_image(path: str | Path, bgr_uint8: np.ndarray, *, format_hint: Optional[str] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = (format_hint or p.suffix).lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        cv2.imwrite(str(p), bgr_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == "png":
        cv2.imwrite(str(p), bgr_uint8, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    elif ext == "webp":
        cv2.imwrite(str(p), bgr_uint8, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
    else:
        cv2.imwrite(str(p), bgr_uint8)


def apply_canvas_texture_overlay(
    out_bgr_uint8: np.ndarray,
    canvas_texture_bgr_uint8: np.ndarray,
    *,
    strength: float = 0.12,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Subtle texture overlay:
      out = out * (1 + strength * tex_centered)
    where tex_centered is texture grayscale mapped from [0,1] to [-1,1].
    """
    if out_bgr_uint8 is None or canvas_texture_bgr_uint8 is None:
        return out_bgr_uint8

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return out_bgr_uint8

    h, w = out_bgr_uint8.shape[:2]
    tex = cv2.resize(canvas_texture_bgr_uint8, (w, h), interpolation=cv2.INTER_AREA)
    tex_gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if float(blur_sigma) > 1e-6:
        tex_gray = cv2.GaussianBlur(tex_gray, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    tex_centered = (tex_gray - 0.5) * 2.0  # [-1, 1]
    mod = 1.0 + strength * tex_centered
    mod = np.clip(mod, 0.0, 2.0)[:, :, None]

    out = out_bgr_uint8.astype(np.float32) * mod
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class ResizeInfo:
    scale: float
    orig_size: Tuple[int, int]  # (w, h)
    new_size: Tuple[int, int]  # (w, h)


def resize_max_side(bgr_uint8: np.ndarray, max_side: int) -> tuple[np.ndarray, ResizeInfo]:
    h, w = bgr_uint8.shape[:2]
    max_side = int(max(1, max_side))
    m = max(h, w)
    if m <= max_side:
        info = ResizeInfo(scale=1.0, orig_size=(w, h), new_size=(w, h))
        return bgr_uint8, info

    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr_uint8, (nw, nh), interpolation=cv2.INTER_AREA)
    info = ResizeInfo(scale=scale, orig_size=(w, h), new_size=(nw, nh))
    return resized, info


def resize_to(bgr_uint8: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = int(size_wh[0]), int(size_wh[1])
    return cv2.resize(bgr_uint8, (w, h), interpolation=cv2.INTER_CUBIC)


def make_error_heatmap_bgr(err_f32: np.ndarray) -> np.ndarray:
    e = err_f32.astype(np.float32)
    e = e - float(e.min())
    denom = float(e.max()) if float(e.max()) > 1e-8 else 1.0
    e = (255.0 * (e / denom)).clip(0.0, 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(e, cv2.COLORMAP_TURBO)
    return heat


def encode_image_bytes(bgr_uint8: np.ndarray, fmt: str = "png") -> bytes:
    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg", "webp"}:
        fmt = "png"
    ext = ".jpg" if fmt == "jpeg" else f".{fmt}"
    ok, buf = cv2.imencode(ext, bgr_uint8)
    if not ok:
        raise RuntimeError(f"Failed encoding image as {fmt}")
    return bytes(buf)


def zip_images(
    items: Iterable[tuple[str, np.ndarray]],
    *,
    image_format: str = "png",
    zip_name: str = "painterly_results.zip",
) -> Path:
    out_path = Path.cwd() / zip_name
    image_format = image_format.lower().lstrip(".")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, bgr in items:
            safe_name = Path(name).name
            stem = Path(safe_name).stem or "image"
            ext = "jpg" if image_format in {"jpg", "jpeg"} else image_format
            arc = f"{stem}.{ext}"
            data = encode_image_bytes(bgr, fmt=image_format)
            zf.writestr(arc, data)

    return out_path


class SimpleTimer:
    def __init__(self) -> None:
        self._t0 = time.time()

    def reset(self) -> None:
        self._t0 = time.time()

    def elapsed_s(self) -> float:
        return time.time() - self._t0

