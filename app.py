from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
from PIL import Image

from .painter import paintify
from .utils import (
    SimpleTimer,
    apply_canvas_texture_overlay,
    bgr_uint8_to_pil,
    load_image_bgr,
    pil_to_bgr_uint8,
    save_image,
    zip_images,
)


PRESETS: dict[str, dict[str, Any]] = {
    "Custom": {},
    "Hertzmann_1998_Classic_CoarseToFine": {
        "brush_radii": "8,4,2",
        "threshold": 50.0,
        "max_stroke_length": 16,
        "min_stroke_length": 4,
        "curvature": 1.0,
        "opacity": 0.85,
        "grid_factor": 1.0,
    },
    "Hertzmann_1998_Coarse_Underpainting_Fast": {
        "brush_radii": "16,8,4",
        "threshold": 60.0,
        "max_stroke_length": 18,
        "min_stroke_length": 4,
        "curvature": 0.9,
        "opacity": 0.95,
        "grid_factor": 1.2,
    },
    "Hertzmann_1998_ShortStroke_StippleLike": {
        "brush_radii": "4,2,1",
        "threshold": 30.0,
        "max_stroke_length": 6,
        "min_stroke_length": 2,
        "curvature": 0.2,
        "opacity": 0.9,
        "grid_factor": 0.9,
    },
    "Hertzmann_1998_FineDetail_DenseGrid_Slow": {
        "brush_radii": "4,2",
        "threshold": 30.0,
        "max_stroke_length": 14,
        "min_stroke_length": 3,
        "curvature": 0.9,
        "opacity": 0.9,
        "grid_factor": 0.7,
    },
}


def _frame_sort_key(filename: str) -> tuple:
    """
    Sort key for image-sequence filenames (e.g. shot.0042.png, frame_000123.exr).
    Uses the last contiguous digit group in the stem (common for Blender/VSE exports).
    Files without digits sort after numbered ones, then alphabetically.
    """
    stem = Path(filename).stem
    nums = re.findall(r"\d+", stem)
    if nums:
        return (0, int(nums[-1]), stem.lower())
    return (1, 0, stem.lower())


def _parse_radii(text: str) -> list[int]:
    parts = [p.strip() for p in (text or "").replace(" ", "").split(",") if p.strip()]
    if not parts:
        return [8, 4, 2]
    out: list[int] = []
    for p in parts:
        out.append(int(float(p)))
    return out


def _estimate_seconds(width: int, height: int, radii: list[int], max_len: int, grid_factor: float, fast_preview: bool):
    radii2 = sorted(radii, reverse=True)
    if fast_preview and radii2:
        radii2 = [radii2[0]]
    area = float(width * height)
    # Very rough: cells per layer * expected stroke length work.
    work = 0.0
    for r in radii2:
        step = max(1.0, float(r) * float(grid_factor))
        cells = area / (step * step)
        work += cells * float(max_len)
    # Calibrate to “seconds” with a conservative constant.
    return max(0.1, work / 2.5e6)


def _paint_single(
    image: Image.Image,
    brush_radii: str,
    threshold: float,
    max_stroke_length: int,
    min_stroke_length: int,
    curvature: float,
    opacity: float,
    grid_factor: float,
    fast_preview: bool,
    preview_max_side: int,
    time_budget_s: float,
    max_strokes_per_layer: int,
    underpaint: bool,
    underpaint_mode: str,
    force_coverage: bool,
    canvas_texture: Optional[str],
    canvas_texture_opacity: float,
    canvas_texture_blur: float,
    seed: int,
    brush_texture: Optional[str],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if image is None:
        raise gr.Error("Please upload an image.")

    radii = _parse_radii(brush_radii)
    w, h = image.size
    est = _estimate_seconds(w, h, radii, max_stroke_length, grid_factor, fast_preview)

    progress(0, desc=f"Painting (ETA ~{est:.1f}s)…")
    t = SimpleTimer()

    bgr = pil_to_bgr_uint8(image)
    brush_texture_path = None
    if brush_texture:
        brush_texture_path = getattr(brush_texture, "name", None) or str(brush_texture)

    out = paintify(
        bgr,
        brush_radii=radii,
        threshold=threshold,
        max_stroke_length=max_stroke_length,
        min_stroke_length=min_stroke_length,
        curvature=curvature,
        opacity=opacity,
        grid_factor=grid_factor,
        brush_texture_path=brush_texture_path,
        fast_preview=fast_preview,
        seed=seed,
        preview_max_side=preview_max_side,
        time_budget_s=float(time_budget_s) if time_budget_s else None,
        max_strokes_per_layer=int(max_strokes_per_layer) if max_strokes_per_layer else None,
        underpaint=bool(underpaint),
        underpaint_mode=str(underpaint_mode),
        force_coverage=bool(force_coverage),
    )
    if canvas_texture is not None:
        canvas_texture_path = getattr(canvas_texture, "name", None) or str(canvas_texture)
        tex_bgr = load_image_bgr(canvas_texture_path)
        out = apply_canvas_texture_overlay(
            out,
            tex_bgr,
            strength=float(canvas_texture_opacity),
            blur_sigma=float(canvas_texture_blur),
        )
    out_pil = bgr_uint8_to_pil(out)
    elapsed = t.elapsed_s()
    progress(1.0, desc="Done")

    fd, out_path = tempfile.mkstemp(prefix="painterly_", suffix=".png")
    os.close(fd)
    out_pil.save(out_path, format="PNG")

    summary = f"Done in {elapsed:.2f}s (estimated {est:.1f}s)."
    return image, out_pil, summary, out_path


def _process_batch(
    images: list[Image.Image],
    filenames: list[str],
    output_format: str,
    brush_radii: str,
    threshold: float,
    max_stroke_length: int,
    min_stroke_length: int,
    curvature: float,
    opacity: float,
    grid_factor: float,
    fast_preview: bool,
    preview_max_side: int,
    seed: int,
    brush_texture: Optional[str],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if not images:
        raise gr.Error("Please upload one or more images.")

    radii = _parse_radii(brush_radii)
    results: list[Image.Image] = []
    zip_items: list[tuple[str, np.ndarray]] = []

    t = SimpleTimer()
    n = len(images)
    for i, img in enumerate(images):
        progress(i / max(1, n), desc=f"Processing {i+1}/{n}…")
        bgr = pil_to_bgr_uint8(img)
        out = paintify(
            bgr,
            brush_radii=radii,
            threshold=threshold,
            max_stroke_length=max_stroke_length,
            min_stroke_length=min_stroke_length,
            curvature=curvature,
            opacity=opacity,
            grid_factor=grid_factor,
            brush_texture_path=brush_texture,
            fast_preview=fast_preview,
            seed=seed + i,  # deterministic but distinct per image
            preview_max_side=preview_max_side,
        )
        out_pil = bgr_uint8_to_pil(out)
        results.append(out_pil)

        name = filenames[i] if i < len(filenames) else f"image_{i+1}.png"
        zip_items.append((name, out))

    zip_path = zip_images(zip_items, image_format=output_format, zip_name="painterly_batch.zip")
    progress(1.0, desc="Done")

    summary = f"Processed {n} image(s) in {t.elapsed_s():.2f}s."
    return results, str(zip_path), summary


def _export_batch_to_folder(
    images: list[Image.Image],
    filenames: list[str],
    output_dir: str,
    output_format: str,
    overwrite: bool,
    numbered_sequence_names: bool,
    brush_radii: str,
    threshold: float,
    max_stroke_length: int,
    min_stroke_length: int,
    curvature: float,
    opacity: float,
    grid_factor: float,
    fast_preview: bool,
    preview_max_side: int,
    time_budget_s: float,
    max_strokes_per_layer: int,
    underpaint: bool,
    underpaint_mode: str,
    force_coverage: bool,
    canvas_texture: Optional[str],
    canvas_texture_opacity: float,
    canvas_texture_blur: float,
    seed: int,
    brush_texture: Optional[str],
    dedupe_skipped: int = 0,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if not images:
        raise gr.Error("Please upload one or more images.")
    if len(images) > 500:
        raise gr.Error("Batch export is limited to 500 images.")
    if not output_dir or not str(output_dir).strip():
        raise gr.Error("Please enter an output folder path.")

    out_dir = Path(str(output_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    radii = _parse_radii(brush_radii)

    brush_texture_path = None
    if brush_texture:
        brush_texture_path = getattr(brush_texture, "name", None) or str(brush_texture)
    canvas_texture_bgr = None
    if canvas_texture:
        canvas_texture_path = getattr(canvas_texture, "name", None) or str(canvas_texture)
        canvas_texture_bgr = load_image_bgr(canvas_texture_path)

    # Normalize output format.
    fmt = str(output_format).lower().strip().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg", "webp"}:
        fmt = "jpg" if fmt in {"jpeg"} else "png"
    ext = "jpg" if fmt in {"jpg", "jpeg"} else fmt

    processed = 0
    skipped = 0

    def maybe_progress(frac: float, desc: str) -> None:
        try:
            progress(frac, desc=desc)
        except Exception:
            # Allows local invocation outside Gradio queue.
            return

    t = SimpleTimer()
    n = len(images)
    stem_counts: dict[str, int] = {}
    for i, (img, in_name) in enumerate(zip(images, filenames)):
        maybe_progress(i / max(1, n), desc=f"Exporting {i+1}/{n}…")

        if numbered_sequence_names:
            # One output per batch index — avoids stem collisions for image sequences.
            out_path = out_dir / f"frame_{i + 1:06d}_painterly.{ext}"
        else:
            in_stem = Path(in_name).stem
            # Sanitize stem for Windows paths
            safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in in_stem) or "image"
            key = safe.lower()
            stem_counts[key] = stem_counts.get(key, 0) + 1
            dup_n = stem_counts[key]
            if dup_n == 1:
                out_path = out_dir / f"{safe}_painterly.{ext}"
            else:
                out_path = out_dir / f"{safe}_{dup_n}_painterly.{ext}"

        if out_path.exists() and not bool(overwrite):
            skipped += 1
            continue

        bgr = pil_to_bgr_uint8(img)
        out_bgr = paintify(
            bgr,
            brush_radii=radii,
            threshold=threshold,
            max_stroke_length=max_stroke_length,
            min_stroke_length=min_stroke_length,
            curvature=curvature,
            opacity=opacity,
            grid_factor=grid_factor,
            brush_texture_path=brush_texture_path,
            fast_preview=fast_preview,
            seed=seed + i,
            preview_max_side=preview_max_side,
            time_budget_s=float(time_budget_s) if time_budget_s else None,
            max_strokes_per_layer=int(max_strokes_per_layer) if max_strokes_per_layer else None,
            underpaint=bool(underpaint),
            underpaint_mode=str(underpaint_mode),
            force_coverage=bool(force_coverage),
        )
        if canvas_texture_bgr is not None:
            out_bgr = apply_canvas_texture_overlay(
                out_bgr,
                canvas_texture_bgr,
                strength=float(canvas_texture_opacity),
                blur_sigma=float(canvas_texture_blur),
            )

        # Save to disk (no ZIP / no in-memory packaging).
        save_image(out_path, out_bgr, format_hint=ext)
        processed += 1

    maybe_progress(1.0, desc="Export done")
    parts = [f"Exported {processed} image(s), skipped {skipped}. Time: {t.elapsed_s():.2f}s"]
    if dedupe_skipped:
        parts.append(f"Skipped {dedupe_skipped} duplicate upload(s) (same file path).")
    return " ".join(parts)


def _files_to_images_and_names(
    files,
    *,
    sort_by_frame_number: bool = False,
) -> tuple[list[Image.Image], list[str], int]:
    """Load uploaded files; dedupe by resolved path; optional sort for image sequences."""
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    dedupe_skipped = 0
    for f in files or []:
        path = getattr(f, "name", None) or str(f)
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            resolved = str(Path(path))
        if resolved in seen:
            dedupe_skipped += 1
            continue
        seen.add(resolved)
        name = Path(path).name
        entries.append((path, name))

    if sort_by_frame_number:
        entries.sort(key=lambda e: _frame_sort_key(e[1]))

    imgs: list[Image.Image] = []
    names: list[str] = []
    for path, name in entries:
        imgs.append(Image.open(path).convert("RGB"))
        names.append(name)
    return imgs, names, dedupe_skipped


def _export_batch_from_files(
    files,
    output_dir: str,
    output_format: str,
    overwrite: bool,
    numbered_sequence_names: bool,
    sort_by_frame_number: bool,
    brush_radii: str,
    threshold: float,
    max_stroke_length: int,
    min_stroke_length: int,
    curvature: float,
    opacity: float,
    grid_factor: float,
    fast_preview: bool,
    preview_max_side: int,
    time_budget_s: float,
    max_strokes_per_layer: int,
    underpaint: bool,
    underpaint_mode: str,
    force_coverage: bool,
    canvas_texture: Optional[str],
    canvas_texture_opacity: float,
    canvas_texture_blur: float,
    seed: int,
    brush_texture: Optional[str],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    images, names, dedupe_skipped = _files_to_images_and_names(
        files, sort_by_frame_number=bool(sort_by_frame_number)
    )
    return _export_batch_to_folder(
        images=images,
        filenames=names,
        output_dir=output_dir,
        output_format=output_format,
        overwrite=overwrite,
        numbered_sequence_names=bool(numbered_sequence_names),
        brush_radii=brush_radii,
        threshold=threshold,
        max_stroke_length=max_stroke_length,
        min_stroke_length=min_stroke_length,
        curvature=curvature,
        opacity=opacity,
        grid_factor=grid_factor,
        fast_preview=fast_preview,
        preview_max_side=preview_max_side,
        time_budget_s=time_budget_s,
        max_strokes_per_layer=max_strokes_per_layer,
        underpaint=underpaint,
        underpaint_mode=underpaint_mode,
        force_coverage=force_coverage,
        canvas_texture=canvas_texture,
        canvas_texture_opacity=canvas_texture_opacity,
        canvas_texture_blur=canvas_texture_blur,
        seed=seed,
        brush_texture=brush_texture,
        dedupe_skipped=dedupe_skipped,
        progress=progress,
    )


def _preset_to_controls(preset_name: str):
    p = PRESETS.get(preset_name, {})
    return (
        p.get("brush_radii", gr.update()),
        p.get("threshold", gr.update()),
        p.get("max_stroke_length", gr.update()),
        p.get("min_stroke_length", gr.update()),
        p.get("curvature", gr.update()),
        p.get("opacity", gr.update()),
        p.get("grid_factor", gr.update()),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Painterly Stroke Renderer") as demo:
        gr.Markdown(
            "### Painterly Stroke Renderer\n"
            "Multi-layer stroke-based rendering (Hertzmann 1998). Use **Fast Preview** for quick feedback.\n"
        )

        # Sliders/controls at the top
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS.keys()), value="Custom", label="Preset (Hertzmann 1998 variants)")
            brush_tex = gr.File(
                label="Optional brush texture PNG (slower). White=paint, black=transparent.",
                file_types=[".png"],
            )
            canvas_tex = gr.File(
                label="Optional canvas texture image (jpg/png, post-process)",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
            )

        with gr.Row():
            brush_radii = gr.Textbox(
                value="8,4,2",
                label="brush_radii (larger radii=faster, smaller radii=slower)",
            )
            threshold = gr.Slider(
                1.0,
                120.0,
                value=55.0,
                step=1.0,
                label="threshold (higher=faster, lower=slower)",
            )
            grid_factor = gr.Slider(
                0.25,
                3.0,
                value=1.1,
                step=0.05,
                label="grid_factor (higher=faster, lower=slower)",
            )

        with gr.Row():
            max_stroke_length = gr.Slider(
                2,
                64,
                value=14,
                step=1,
                label="max_stroke_length (lower=faster, higher=slower)",
            )
            min_stroke_length = gr.Slider(
                1,
                32,
                value=4,
                step=1,
                label="min_stroke_length (slightly faster at lower values)",
            )
            curvature = gr.Slider(
                0.0,
                1.0,
                value=1.0,
                step=0.05,
                label="curvature (mostly quality; minimal speed impact)",
            )
            opacity = gr.Slider(
                0.05,
                1.0,
                value=0.9,
                step=0.05,
                label="opacity (quality; minimal speed impact)",
            )

        with gr.Row():
            fast_preview = gr.Checkbox(value=True, label="Fast Preview (much faster; largest radius only)")
            preview_max_side = gr.Slider(
                300,
                2400,
                value=900,
                step=50,
                label="preview_max_side (lower=faster, higher=slower)",
            )
            underpaint = gr.Checkbox(value=True, label="Underpaint (fills canvas immediately; recommended)")
            underpaint_mode = gr.Radio(
                ["average", "blur"],
                value="average",
                label="Underpaint mode (average=painterly strokes, blur=photo-like base)",
            )
            force_coverage = gr.Checkbox(value=True, label="Force coverage (fixes white gaps; can be slower)")
            canvas_texture_opacity = gr.Slider(
                0.0,
                0.5,
                value=0.12,
                step=0.01,
                label="canvas_texture_opacity (0=subtle off, higher=stronger texture)",
            )
            canvas_texture_blur = gr.Slider(
                0.0,
                4.0,
                value=0.6,
                step=0.1,
                label="canvas_texture_blur (higher=softer texture)",
            )
            time_budget_s = gr.Slider(
                5,
                60,
                value=30,
                step=1,
                label="time_budget_s (hard cap; lower=faster, higher=slower)",
            )
            max_strokes_per_layer = gr.Slider(
                100,
                20000,
                value=4000,
                step=100,
                label="max_strokes_per_layer (hard cap; lower=faster, higher=slower)",
            )
            seed = gr.Number(value=1, precision=0, label="seed (deterministic; no speed impact)")

        # Input / Output only
        with gr.Row():
            in_img = gr.Image(type="pil", label="Input")
            out_img = gr.Image(type="pil", label="Output")

        with gr.Row():
            paint_btn = gr.Button("Paintify", variant="primary")
            download_single = gr.File(label="Download result")

        status = gr.Markdown("")

        preset.change(
            fn=_preset_to_controls,
            inputs=[preset],
            outputs=[
                brush_radii,
                threshold,
                max_stroke_length,
                min_stroke_length,
                curvature,
                opacity,
                grid_factor,
            ],
        )

        paint_btn.click(
            fn=_paint_single,
            inputs=[
                in_img,
                brush_radii,
                threshold,
                max_stroke_length,
                min_stroke_length,
                curvature,
                opacity,
                grid_factor,
                fast_preview,
                preview_max_side,
                time_budget_s,
                max_strokes_per_layer,
                underpaint,
                underpaint_mode,
                force_coverage,
                canvas_tex,
                canvas_texture_opacity,
                canvas_texture_blur,
                seed,
                brush_tex,
            ],
            outputs=[in_img, out_img, status, download_single],
        )

        # Batch export to disk (up to 500 images)
        gr.Markdown("---")
        gr.Markdown("## Batch Export to Folder (up to 500 images)")

        with gr.Row():
            batch_in = gr.Files(label="Input images (limit: 500)", file_types=["image"])
            output_dir = gr.Textbox(label="Output folder path", value=str(Path.cwd() / "painterly_out"))

        with gr.Row():
            output_format = gr.Radio(["png", "jpg"], value="jpg", label="Output format")
            overwrite = gr.Checkbox(value=True, label="Overwrite existing files")
            batch_numbered_sequence = gr.Checkbox(
                value=True,
                label="Sequence filenames (frame_000001_painterly.* … avoids duplicate stems)",
            )
            batch_sort_by_frame = gr.Checkbox(
                value=True,
                label="Sort by number in filename (fixes Blender jumps — upload order is often wrong)",
            )

        with gr.Row():
            batch_btn = gr.Button("Export Batch", variant="primary")
            batch_status = gr.Markdown("")

        batch_btn.click(
            fn=_export_batch_from_files,
            inputs=[
                batch_in,
                output_dir,
                output_format,
                overwrite,
                batch_numbered_sequence,
                batch_sort_by_frame,
                brush_radii,
                threshold,
                max_stroke_length,
                min_stroke_length,
                curvature,
                opacity,
                grid_factor,
                fast_preview,
                preview_max_side,
                time_budget_s,
                max_strokes_per_layer,
                underpaint,
                underpaint_mode,
                force_coverage,
                canvas_tex,
                canvas_texture_opacity,
                canvas_texture_blur,
                seed,
                brush_tex,
            ],
            outputs=[batch_status],
        )

        demo.queue()
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

