# Painterly Stroke Renderer (Gradio)

Local Python web app that turns photos into painterly, stroke-based renderings using Aaron Hertzmann’s SIGGRAPH 1998 algorithm (multi-layer coarse→fine curved brush strokes).

## Setup

From `C:\Cursor Projects`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\painterly_app\requirements.txt
```

## Run

```powershell
python -m painterly_app.app
```

Open the local Gradio URL it prints.

## Notes / troubleshooting

- If results look like a blur, raise **threshold** a bit and/or increase **grid_factor** (fewer strokes), and make sure **Fast Preview** is off for final renders.
- Large images can take time. The preview tab downsamples any side larger than `preview_max_side` and upsamples the result back.
- For more “oil paint” texture, upload a brush PNG in the UI. See `painterly_app/brushes/README.txt`.

### Batch export + Blender / image sequences

Gradio often returns **multi-file uploads in arbitrary order**. If you export as `frame_000001_painterly.*`, `frame_000002_painterly.*`, … those numbers may **not** match time order unless you sort by filename. Leave **“Sort by number in filename”** **ON** when sending a sequence to Blender’s VSE — otherwise playback can jump or look like repeated frames.

If consecutive frames still *look* identical, try turning **Fast Preview** off for the batch and/or raising **time_budget_s** so each frame gets more strokes.

