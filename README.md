# AI Powered Virtual Mouse (Windows)

Use hand gestures to control mouse, scroll, drag & drop, image zoom/rotate/flip, system volume, and screen brightness using a webcam.

## Requirements
- Windows 10/11 (PowerShell)
- Python 3.10–3.12 (64-bit recommended)
- A webcam

## Install
1. Create and activate a virtual environment (recommended):
   - PowerShell:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

If `mediapipe` build fails on older Python, install Python 3.10–3.12.

## Run
```powershell
python VirtualMouse.py
```
- Press `q` to quit.

## Gestures
- Move cursor: Raise only index finger and move.
- Left click: Pinch index + thumb briefly (keep middle down).
- Right click: Pinch middle + thumb when index is up.
- Drag & drop: Keep index + thumb pinched to hold; release to drop.
- Scroll: Raise index + middle; move up/down to scroll.
- Zoom image: Change distance between index and thumb.
- Rotate image: Rotate the vector between index and thumb.
- Flip image: Raise only pinky (debounced toggle).
- Volume: Raise only ring finger; move up to increase, down to decrease.
- Brightness: Raise all fingers; move up to increase, down to decrease.

Notes:
- Volume control requires `pycaw`; if unavailable, it will be skipped.
- Brightness control requires `screen-brightness-control`; if unavailable, it will be skipped.
- A small demo image panel is shown in the top-left to visualize zoom/rotate/flip.

## Troubleshooting
- If the mouse moves too fast or jittery, tune `SMOOTHING`, `CLICK_DIST_PX`, and `DRAG_DIST_PX` constants in `VirtualMouse.py`.
- If brightness/volume do not change, they may require admin permissions or compatible hardware/drivers.
- If your webcam opens but tracking is poor, ensure good lighting and keep your hand fully in frame.
