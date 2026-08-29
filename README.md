# Transparent Cover Mockup

Desktop app for print-ready mockups of **transparent phone covers**. Upload a cover image and a design; the design is placed only on the flat back panel. Camera cutouts, bumper, rim, and side walls stay clear.

Version 1 is offline, local, and focused on this core workflow. No accounts, no internet, no ML APIs.

## Requirements

- Python 3.11+
- Windows, macOS, or Linux

## Install

```bash
cd cover_software
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The app does not ship built-in cover photos. Upload your own cover and design. Sample PNGs used by `--smoke-test` are generated locally on first run.

## Workflow

1. Click **Upload Cover** and choose a transparent phone-cover PNG (JPG is supported when the backdrop is plain).
2. Click **Upload Design** and choose a PNG, JPG, JPEG, or WebP.
3. The software detects the back panel, camera area, and bumper, then fits the design (aspect ratio preserved, never stretched).
4. The mockup shows original artwork colors — the cover is not blended as a gray haze over the print area.
5. **Export PNG** (keeps transparency) or **Export JPG**.

**Reset** clears both images.

## How printing is constrained

Artwork is never resized over the whole cover image. A dedicated mask pipeline is used:

```
FINAL_PRINT_MASK =
    BACK_PRINT_AREA
    minus CAMERA_EXCLUSION_AREA
    minus EDGE / BUMPER AREA
```

Masks are antialiased so corners stay rounded and edges do not stair-step or halo.

Artwork is composited **inside the printable mask only**. Cover pixels are kept for the camera module, bumper, and background — they are not used as a full-image white/gray overlay.

## Debug mode

In `app/utils/constants.py`:

```python
DEBUG_MODE = False
```

Set this to `True` to show detection-mask thumbnails (contour, printable area, camera, edges) in the window.

## Adding another phone cover later

Upload any cover PNG. Geometry is detected from the image (alpha, contours, camera holes). `app/core/templates.py` is only used to generate the sample PNG for automated tests.

## Project layout

```
cover_software/
  main.py
  requirements.txt
  app/ui/main_window.py          GUI only
  app/core/cover_processor.py    pipeline
  app/core/cover_detector.py     outer / back-panel detection
  app/core/camera_detector.py    camera cutout / island
  app/core/mask_generator.py     printable / camera / edge masks
  app/core/image_transform.py    fit / crop / perspective
  app/core/compositor.py         no-haze compositing
  app/core/export_manager.py     PNG / JPG
  assets/covers/                 empty (samples generated locally)
  output/
```

## Headless check

```bash
python main.py --smoke-test
```

Confirms the design is applied inside the printable area and that camera / bumper pixels stay equal to the original cover.
