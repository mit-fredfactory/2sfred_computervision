# FrED Computer Vision Lab: Measuring a Fiber with a Camera

FrED measures fiber diameter with a small USB camera. In this lab you connect that
camera to your laptop and walk through the CV techniques
one step at a time. You will predict what each step does, see it on the live image,
and write a few short pieces of code.

## Rules for the hardware

| Knob | You may... |
|---|---|
| **Light intensity (0-100 %)** | change it whenever an exercise asks you to |
| **Focus** | **leave it alone.** If you change it, the pixel-to-mm calibration is no longer valid |

The wire and the camera are fixed. The scene is always a **vertical white wire on a black card**.
When an exercise needs a rotated or shifted image, the code does it in software.

## Setup (10 min before the lab, Windows or macOS)

1. Install VS Code on your laptop.
2. Install Python 3.9 or newer, then:
   ```
   pip install opencv-python numpy
   ```
3. Plug FrED's camera USB cable into your laptop.
4. Find the camera index. In the `lab` folder, run:
   ```
   python labkit.py --list
   ```
   Open the thumbnails it saves in `lab/snapshots/`. The FrED camera shows a white wire on black.
   Use that number as `--index` in every exercise. It is often **1** on Windows. On a Mac it may be 0, 1 or 2,
   because the built-in camera and an iPhone (Continuity Camera) also take index numbers.
5. If the camera does not open:
   - **Windows:** close the Camera app, Teams, Zoom, etc.
   - **macOS:** go to System Settings → Privacy & Security → Camera and enable Terminal
     (or VS Code / PyCharm). Then **quit and restart** that app. Also quit Photo Booth / FaceTime.
   - No camera at all? Every script also runs on a saved frame (`--image frames/sample_live.png`) or on a
     computer-generated wire (`--synthetic`, which also has a working light slider).

## The lab window

Every exercise opens one window. It shows a row of panels, the readings, and the controls.

| Key | What it does |
|---|---|
| `space` | freeze / unfreeze the image (sliders still work on a frozen image) |
| `f` | move the yellow focus box to the next panel |
| `b` | remember the focused panel as **BEFORE** |
| `d` | show **BEFORE / NOW / difference** for the focused panel |
| `z` | zoom into the wire's left edge |
| `s` | save a snapshot to `snapshots/`; compare them later with `compare.py` |
| `q` | quit |
| other letters | toggles listed at the bottom of each window |

**How to make a before/after comparison:** press `f` until the panel you care about is boxed, then press `b`.
Change a slider, toggle, or knob, and press `d`.

**How to compare saved snapshots:** `python compare.py --latest 4` (add `--panel N` to show only one panel of each).

**How to check your code:** `python check.py ex2` (or `python check.py` to check everything).

## Exercises

✏️ = you write code (1-2 lines each) · 👀 = predict, then observe

| # | Topic | Slides | You do | Min |
|---|---|---|---|---|
| 0 | `ex0_live.py`: FrED's pipeline, live | 4 | 👀 read the diameter in px; switch FrED's filters on and off | 5 |
| 1 | `ex1_otsu.py`: pixels, histogram, Otsu | 6-17 | ✏️ Otsu threshold · 👀 light knob at 100/50/20/0 % | 15 |
| 2 | `ex2_kernels.py`: Sobel and smoothing | 19-27 | ✏️ `KY`, `BOX`, gradient magnitude · 👀 rotate, noise, blur, flip | 20 |
| 3 | `ex3_morphology.py`: erode, dilate, opening | 30-33 | 👀 predict the width change; remove dust | 10 |
| 4 | `ex4_hough.py`: Canny → Hough → diameter | 28, 34-36 | 👀 Hough and Canny sliders · ✏️ `boundary()` instead of Canny | 15 |
| 5 | `ex5_template.py`: where is the fiber? | 23-26 | ✏️ `find_template()` · 👀 shift, dim the light | 10 |
| 6 | `ex6_calibration.py`: pixels → mm | 70 | ✏️ coefficient, `px_to_mm` | 10 |

Each file starts with step-by-step instructions and questions. Open it in your editor next to the window.
Run each exercise with:

```
python ex1_otsu.py --index 1
```

## What to hand in

1. The output of `python check.py` (all PASS).
2. Four `compare.py` screenshots, saved with `--out`:
   - Ex 1: fixed T vs Otsu at light 100 / 50 / 20 / 0 %
   - Ex 2: Kx with noise, before and after the box blur
   - Ex 3: dust specks, before and after the opening
   - Ex 6: your calibrated reading in mm
3. Short answers to the questions at the top of each exercise file.

