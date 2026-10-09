"""Shared helpers for the FrED computer-vision lab (Windows and macOS).

You do not need to edit this file.

    python labkit.py --list                  which camera index is the FrED camera?
    python labkit.py --capture frames/x.png  save one camera frame

Every exercise script accepts the same options:
    --index N        camera index (default 1)
    --image FILE     use a saved frame instead of the camera
    --synthetic      use a computer-generated wire (no camera needed)

Keys in every lab window:
    space  freeze / unfreeze the frame (sliders still work on a frozen frame)
    f      move the yellow focus box to the next panel
    b      remember the focused panel as BEFORE
    d      show BEFORE | NOW | difference for the focused panel
    z      zoom into the wire's left edge
    s      save a snapshot to snapshots/ (use compare.py to view them)
    q      quit
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

LAB_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = LAB_DIR / "snapshots"
DEFAULT_INDEX = 1
PANEL_W, PANEL_H = 384, 288
TITLE_H = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX
RESERVED_KEYS = set(" fbdzsq")


# ----------------------------------------------------------------- camera

def backend() -> int:
    """MSMF on Windows (DirectShow cannot open the FrED camera), AVFoundation on macOS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_MSMF
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def camera_help() -> str:
    if platform.system() == "Darwin":
        return ("- run `python labkit.py --list` to find the right --index\n"
                "- allow camera access: System Settings > Privacy & Security > Camera >\n"
                "  enable Terminal (or VS Code / PyCharm), then restart that app\n"
                "- quit Photo Booth, FaceTime, Zoom or any other app using the camera")
    return ("- run `python labkit.py --list` to find the right --index\n"
            "- close the Windows Camera app, Teams, Zoom or any other app using the camera\n"
            "- unplug and replug the USB cable")


def open_camera(index: int) -> cv2.VideoCapture:
    """Open camera `index` and wait until it delivers real frames."""
    capture = cv2.VideoCapture(index, backend())
    if capture.isOpened():
        for _ in range(40):  # the first frames can be empty while the camera starts
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                return capture
            time.sleep(0.05)
    capture.release()
    raise SystemExit(f"Could not get frames from camera index {index}.\n{camera_help()}")


def list_cameras(max_index: int = 5) -> None:
    """Try indices 0..max_index-1 and save a thumbnail of each one that works."""
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    found = False
    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend())
        frame = None
        if capture.isOpened():
            for _ in range(20):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
                time.sleep(0.05)
        capture.release()
        if frame is None:
            print(f"index {index}: no camera")
            continue
        found = True
        path = SNAPSHOT_DIR / f"camera_index_{index}.png"
        cv2.imwrite(str(path), frame)
        print(f"index {index}: {frame.shape[1]}x{frame.shape[0]}, mean brightness "
              f"{frame.mean():5.1f}  -> {path}")
    if found:
        print("\nOpen the thumbnails: the FrED camera shows a white wire on black. "
              "Use that number with --index.")
    else:
        print(camera_help())


def synthetic_frame(light: float = 100, width_px: float = 148, center_x: float = 380,
                    noise: float = 2.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """A computer-generated FrED frame: a vertical white wire on a black card (BGR)."""
    rng = rng if rng is not None else np.random.default_rng()
    h, w = 480, 640
    x = np.arange(w, dtype=np.float32)
    profile = 1 / (1 + np.exp(np.clip((np.abs(x - center_x) - width_px / 2) / 0.8, -50, 50)))
    img = np.tile(2 + 239 * light / 100 * profile, (h, 1))
    img += rng.normal(0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class Source:
    """Frames from the camera, a saved image, or a synthetic wire."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.capture = None
        self.image = None
        self.synthetic = args.synthetic
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists() and (LAB_DIR / image_path).exists():
                image_path = LAB_DIR / image_path
            self.image = cv2.imread(str(image_path))
            if self.image is None:
                raise SystemExit(f"Could not read image {args.image}")
        elif not self.synthetic:
            self.capture = open_camera(args.index)

    def read(self, light: float = 100) -> np.ndarray:
        if self.image is not None:
            return self.image.copy()
        if self.synthetic:
            return synthetic_frame(light)
        for _ in range(20):
            ok, frame = self.capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        raise SystemExit(f"The camera stopped sending frames.\n{camera_help()}")

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def parse_args(description: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX,
                        help=f"camera index (default {DEFAULT_INDEX}); find it with labkit.py --list")
    parser.add_argument("--image", type=Path, help="use a saved frame instead of the camera")
    parser.add_argument("--synthetic", action="store_true", help="use a computer-generated wire")
    parser.add_argument("--smoke", type=int, default=0, help=argparse.SUPPRESS)  # self-test
    return parser.parse_args(argv)


# ----------------------------------------------------------------- drawing

def to_bgr(img) -> np.ndarray:
    """Turn a gray / bool / float / BGR array into a displayable uint8 BGR image."""
    img = np.asarray(img)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img)


def put_text(img, text, org, color=(255, 255, 255), scale=0.45, thickness=1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def signed(img: np.ndarray, full_scale: float) -> np.ndarray:
    """Signed filter output: positive = red, negative = blue, 0 = black."""
    v = np.clip(img / full_scale, -1, 1)
    out = np.zeros(img.shape + (3,), np.uint8)
    out[..., 2] = (np.maximum(v, 0) * 255).astype(np.uint8)
    out[..., 0] = (np.maximum(-v, 0) * 255).astype(np.uint8)
    return out


def magnitude(img: np.ndarray, full_scale: float) -> np.ndarray:
    """Non-negative float image -> gray uint8, `full_scale` maps to white."""
    return (np.clip(img / full_scale, 0, 1) * 255).astype(np.uint8)


def histogram(gray: np.ndarray, marks=(), log: bool = True) -> np.ndarray:
    """Histogram of a uint8 image; `marks` = [(value, (b, g, r), label), ...]."""
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    heights = np.log1p(counts) if log else counts
    w, h, pad = 512, 300, 24
    img = np.full((h, w, 3), 30, np.uint8)
    top = heights.max() or 1
    for value, height in enumerate(heights):
        x = value * 2
        y = int((h - pad) - height / top * (h - 2 * pad))
        cv2.rectangle(img, (x, y), (x + 1, h - pad), (200, 200, 200), -1)
    for i, (value, color, label) in enumerate(marks):
        x = int(round(value)) * 2
        cv2.line(img, (x, pad), (x, h - pad), color, 2)
        put_text(img, f"{label}={value:.0f}", (min(x + 4, w - 90), 16 + 16 * i), color)
    for value in (0, 128, 255):
        put_text(img, str(value), (min(value * 2, w - 28), h - 6), (160, 160, 160), 0.4)
    return img


def patch_view(gray: np.ndarray, x0: int, y0: int, rows: int = 7, cols: int = 11) -> np.ndarray:
    """Draw a rows x cols patch of a gray image as a table of numbers."""
    cell_w, cell_h = 34, 26
    img = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for r in range(rows):
        for c in range(cols):
            y, x = y0 + r, x0 + c
            if not (0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]):
                continue
            v = int(gray[y, x])
            p0, p1 = (c * cell_w, r * cell_h), ((c + 1) * cell_w - 1, (r + 1) * cell_h - 1)
            cv2.rectangle(img, p0, p1, (v, v, v), -1)
            color = (0, 0, 0) if v > 120 else (255, 255, 255)
            cv2.putText(img, str(v), (p0[0] + 3, p0[1] + 18), FONT, 0.42, color, 1, cv2.LINE_AA)
    return img


def draw_lines(bgr: np.ndarray, lines, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    out = to_bgr(bgr).copy()
    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 1, 4)
        for x0, y0, x1, y1 in lines[:, 0]:
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    return out


def line_plot(values, size=(512, 240), y_max=None, color=(0, 220, 255)) -> np.ndarray:
    """A small plot of a list of numbers (for example, calibration samples)."""
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    if not len(values):
        put_text(img, "no samples yet", (10, h // 2))
        return img
    top = y_max or max(max(values), 1)
    step = (w - 20) / max(len(values) - 1, 1)
    pts = [(int(10 + i * step), int(h - 20 - v / top * (h - 40))) for i, v in enumerate(values)]
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, color, -1)
    cv2.line(img, (10, h - 20), (w - 10, h - 20), (120, 120, 120), 1)
    put_text(img, f"{top:.0f}", (4, 16), (160, 160, 160), 0.4)
    put_text(img, "0", (4, h - 6), (160, 160, 160), 0.4)
    return img


# ----------------------------------------------------------------- the lab window

class Lab:
    """A live window that shows the panels returned by an exercise's process(frame)."""

    def __init__(self, title: str, *, light_slider: bool = False, argv=None) -> None:
        self.args = parse_args(title, argv)
        self.title = title
        self.window = title
        self.source = Source(self.args)
        self.sliders: dict[str, int] = {}          # name -> lowest value
        self.toggles: dict[str, list] = {}         # key -> [name, state]
        self.actions: dict[str, tuple] = {}        # key -> (name, callback)
        self.readings: dict[str, str] = {}
        self.notes: list[str] = []
        self.frame = None
        self.frozen = None
        self.before = None
        self.compare = False
        self.zoom = False
        self.zoom_at = (0.5, 0.5)  # zoom center, as a fraction of panel width / height
        self.focus = 1
        self._panels = []
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        if light_slider:
            self.slider("light %", 0, 100, 100)

    # --- controls
    def slider(self, name: str, lo: int, hi: int, init: int) -> None:
        self.sliders[name] = lo
        cv2.createTrackbar(name, self.window, init - lo, hi - lo, lambda _: None)

    def value(self, name: str) -> int:
        return self.sliders[name] + cv2.getTrackbarPos(name, self.window)

    def toggle(self, key: str, name: str, on: bool = False) -> None:
        assert key not in RESERVED_KEYS, f"key {key!r} is reserved"
        self.toggles[key] = [name, on]

    def on(self, name: str) -> bool:
        return any(state for n, state in self.toggles.values() if n == name)

    def action(self, key: str, name: str, callback) -> None:
        assert key not in RESERVED_KEYS, f"key {key!r} is reserved"
        self.actions[key] = (name, callback)

    def reading(self, name: str, value) -> None:
        self.readings[name] = value if isinstance(value, str) else f"{value:.1f}"

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def todo(self, fn, *args, default=None):
        """Call a student function; if it is not written yet, show a note instead of crashing."""
        try:
            return fn(*args)
        except NotImplementedError as error:
            self.note(f"TODO {error}: not written yet")
            return default

    @property
    def light(self) -> int | None:
        return self.value("light %") if "light %" in self.sliders else None

    # --- main loop
    def run(self, process) -> None:
        loops = 0
        try:
            while True:
                if self.frozen is None:
                    self.frame = self.source.read(self.light if self.light is not None else 100)
                frame = self.frozen if self.frozen is not None else self.frame
                self._aim_zoom(frame)
                self.readings, self.notes = {}, []
                if self.light is not None:
                    self.reading("light", f"{self.light} %")
                self._panels = [self._as_panel(p) for p in process(frame.copy())]
                self.focus %= len(self._panels)
                cv2.imshow(self.window, self._compose())
                key = cv2.waitKey(30 if self.frozen is not None else 1) & 0xFF
                loops += 1
                if self.args.smoke and loops >= self.args.smoke:
                    self.snapshot()
                    break
                if not self._handle(key) or self._closed():
                    break
        finally:
            self.source.release()
            cv2.destroyAllWindows()

    def _aim_zoom(self, frame: np.ndarray) -> None:
        """Point the zoom at the wire's left edge (first bright column)."""
        columns = frame.mean(axis=(0, 2)) if frame.ndim == 3 else frame.mean(axis=0)
        bright = np.flatnonzero(columns > (columns.min() + columns.max()) / 2)
        if bright.size and columns.max() - columns.min() > 20:
            self.zoom_at = (bright[0] / frame.shape[1], 0.5)

    def _closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _handle(self, key: int) -> bool:
        if key == 255:
            return True
        if key == 27:
            return False
        char = chr(key).lower()
        if char == "q":
            return False
        if char == " ":
            self.frozen = None if self.frozen is not None else self.frame.copy()
        elif char == "f":
            self.focus = (self.focus + 1) % len(self._panels)
        elif char == "b":
            title, img, _ = self._panels[self.focus]
            self.before = (title, img.copy(), dict(self.readings))
            print(f"BEFORE = '{title}'  {self._readings_text(self.readings)}")
        elif char == "d":
            self.compare = not self.compare and self.before is not None
        elif char == "z":
            self.zoom = not self.zoom
        elif char == "s":
            self.snapshot()
        elif char in self.toggles:
            self.toggles[char][1] = not self.toggles[char][1]
        elif char in self.actions:
            self.actions[char][1]()
        return True

    # --- drawing
    @staticmethod
    def _as_panel(panel):
        title, img, zoomable = (tuple(panel) + (True,))[:3]
        return title, (None if img is None else to_bgr(img)), zoomable

    def _fit(self, img, zoomable):
        if self.zoom and zoomable:
            h, w = img.shape[:2]
            zw, zh = max(w // 4, 8), max(int(w // 4 * PANEL_H / PANEL_W), 6)
            cx, cy = int(self.zoom_at[0] * w), int(self.zoom_at[1] * h)
            x0 = int(np.clip(cx - zw // 2, 0, max(w - zw, 0)))
            y0 = int(np.clip(cy - zh // 2, 0, max(h - zh, 0)))
            img = img[y0:y0 + zh, x0:x0 + zw]
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_AREA
        h, w = img.shape[:2]
        scale = min(PANEL_W / w, PANEL_H / h)
        if scale > 1 and interp == cv2.INTER_AREA:
            interp = cv2.INTER_NEAREST
        return cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=interp)

    def _visible_panels(self):
        if self.compare and self.before is not None:
            b_title, b_img, b_readings = self.before
            n_title, n_img, zoomable = self._panels[self.focus]
            if n_img is None:
                return [(f"BEFORE: {b_title}", b_img, zoomable), (f"NOW: {n_title}", None, False)], b_readings
            if n_img.shape != b_img.shape:
                n_img = cv2.resize(n_img, (b_img.shape[1], b_img.shape[0]))
            diff = cv2.convertScaleAbs(cv2.absdiff(b_img, n_img), alpha=3)
            return [(f"BEFORE: {b_title}", b_img, zoomable), (f"NOW: {n_title}", n_img, zoomable),
                    ("|before - now| x3", diff, zoomable)], b_readings
        return self._panels, None

    def _compose(self) -> np.ndarray:
        panels, before_readings = self._visible_panels()
        fitted = []
        for title, img, zoomable in panels:
            if img is None:
                tile = np.full((PANEL_H // 2, PANEL_W, 3), 60, np.uint8)
                put_text(tile, "TODO: write this function", (20, PANEL_H // 4), (0, 220, 255))
            else:
                tile = self._fit(img, zoomable)
            fitted.append((title, tile))
        n = len(fitted)
        cols = 2 if n == 4 else min(n, 3)
        rows = -(-n // cols)
        cell_h = max(t.shape[0] for _, t in fitted) + TITLE_H
        grid_w = max(cols * PANEL_W, 1000)
        grid = np.full((rows * cell_h, grid_w, 3), 20, np.uint8)
        for i, (title, tile) in enumerate(fitted):
            r, c = divmod(i, cols)
            x, y = c * PANEL_W, r * cell_h
            grid[y + TITLE_H:y + TITLE_H + tile.shape[0], x:x + tile.shape[1]] = tile
            put_text(grid, title[:52], (x + 4, y + 16))
            if not self.compare and i == self.focus:
                cv2.rectangle(grid, (x, y), (x + PANEL_W - 1, y + cell_h - 1), (0, 220, 255), 1)
        return np.vstack([grid, self._status(grid_w, before_readings)])

    @staticmethod
    def _readings_text(readings) -> str:
        return "   ".join(f"{k}: {v}" for k, v in readings.items())

    def _status(self, width: int, before_readings) -> np.ndarray:
        lines = []
        if before_readings is not None:
            lines.append(("BEFORE  " + self._readings_text(before_readings), (255, 200, 120)))
            lines.append(("NOW     " + self._readings_text(self.readings), (120, 255, 120)))
        elif self.readings:
            lines.append((self._readings_text(self.readings), (120, 255, 120)))
        state = [f"[{k}] {n}: {'ON' if s else 'off'}" for k, (n, s) in self.toggles.items()]
        state += [f"[{k}] {n}" for k, (n, _) in self.actions.items()]
        if state:
            lines.append(("   ".join(state), (255, 255, 255)))
        flags = [s for s, on in (("FROZEN", self.frozen is not None), ("ZOOM", self.zoom),
                                 ("BEFORE/NOW", self.compare)) if on]
        lines.append(("space freeze  f focus  b set before  d before/now  z zoom  s snapshot  q quit"
                      + ("    << " + " ".join(flags) + " >>" if flags else ""), (170, 170, 170)))
        lines += [(n, (0, 220, 255)) for n in self.notes]
        bar = np.full((10 + 20 * len(lines), width, 3), 0, np.uint8)
        for i, (text, color) in enumerate(lines):
            put_text(bar, text[: width // 7], (8, 22 + 20 * i), color)
        return bar

    # --- snapshots
    def snapshot(self) -> Path:
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        script = Path(sys.argv[0]).stem
        light = f"_light{self.light:03d}" if self.light is not None else ""
        stem = f"{script}{light}_{time.strftime('%Y%m%d-%H%M%S')}"
        base = SNAPSHOT_DIR / stem
        cv2.imwrite(f"{base}.png", self._compose())
        cv2.imwrite(f"{base}__raw.png", self.frozen if self.frozen is not None else self.frame)
        titles = []
        for i, (title, img, _) in enumerate(self._panels):
            titles.append(title)
            if img is not None:
                cv2.imwrite(f"{base}__p{i}.png", img)
        info = {
            "script": script, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "light": self.light,
            "readings": self.readings, "panels": titles,
            "sliders": {n: self.value(n) for n in self.sliders},
            "toggles": {n: s for n, s in self.toggles.values()},
        }
        Path(f"{base}.json").write_text(json.dumps(info, indent=2))
        print(f"saved snapshot {base}.png   {self._readings_text(self.readings)}")
        return base


# ----------------------------------------------------------------- instructor answers

def use_answers(namespace: dict) -> None:
    """Instructor only: if LAB_ANSWERS names a folder (e.g. lab/solutions), replace this
    exercise's TODO functions and constants with the ones from the same-named file there."""
    folder = os.environ.get("LAB_ANSWERS")
    if not folder:
        return
    folder_path = Path(folder)
    if not folder_path.is_absolute() and not folder_path.exists() and (LAB_DIR / folder_path).exists():
        folder_path = LAB_DIR / folder_path
    path = folder_path.resolve() / Path(namespace["__file__"]).name
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(f"answers_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in getattr(module, "ANSWERS", ()):
        namespace[name] = getattr(module, name)


# ----------------------------------------------------------------- command line

def main() -> None:
    parser = argparse.ArgumentParser(description="FrED lab camera tools")
    parser.add_argument("--list", action="store_true", help="find the FrED camera's index")
    parser.add_argument("--capture", type=Path, help="save one frame from --index to this file")
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX)
    args = parser.parse_args()
    if args.capture:
        capture = open_camera(args.index)
        for _ in range(10):  # let auto-exposure settle
            capture.read()
        ok, frame = capture.read()
        capture.release()
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.capture), frame)
        print(f"saved {args.capture}  (mean brightness {frame.mean():.1f})")
    else:
        list_cameras()


if __name__ == "__main__":
    main()
