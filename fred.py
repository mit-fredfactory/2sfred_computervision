"""The real FrED measurement code (fiber_camera.py), usable without PyQt5 or a database.

fiber_camera.py is not modified. Its imports of PyQt5 and `database` are replaced by
small stand-ins (the same trick as run_usb_camera.py), so we can call its real methods:

    edges, binary = fred.get_edges(frame_rgb)     # gray > erode > dilate > blur > T=100 > Canny
    lines = fred.hough(edges)                     # cv2.HoughLinesP with FrED's settings
    px = fred.diameter_px(lines)                  # FrED's get_fiber_diameter, in pixels
    px, lines, edges, binary = fred.measure(frame_bgr)   # the whole camera_loop
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def _install_stubs() -> None:
    if "fiber_camera" in sys.modules:
        return

    class _Database:
        camera_timestamps, diameter_readings, diameter_setpoint, diameter_delta_time = [], [], [], []

        @staticmethod
        def get_calibration_data(key):
            return 1.0

        @staticmethod
        def update_calibration_data(key, value):
            print(f"[stub Database] {key} = {value} (not saved)")

    database = types.ModuleType("database")
    database.Database = _Database
    sys.modules["database"] = database

    try:  # PyQt5 is only needed for FrED's own GUI; the lab does not use it
        import PyQt5.QtWidgets  # noqa: F401
    except ImportError:
        widgets = types.ModuleType("PyQt5.QtWidgets")
        widgets.QWidget, widgets.QLabel, widgets.QDoubleSpinBox = object, object, object
        gui = types.ModuleType("PyQt5.QtGui")
        gui.QImage, gui.QPixmap = object, object
        core = types.ModuleType("PyQt5.QtCore")
        core.pyqtSignal = lambda *a, **k: None
        sys.modules.update({"PyQt5": types.ModuleType("PyQt5"), "PyQt5.QtWidgets": widgets,
                            "PyQt5.QtGui": gui, "PyQt5.QtCore": core})
    sys.path.insert(0, str(ROOT))


_install_stubs()
import fiber_camera  # noqa: E402

FiberCamera = fiber_camera.FiberCamera

# Hough settings used by FrED (fiber_camera.py, camera_loop)
HOUGH_RHO, HOUGH_THETA, HOUGH_VOTES = 1, np.pi / 180, 30
HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 30, 100
FRED_T = 100  # fixed threshold in get_edges


def camera(erode=True, dilate=True, gaussian=True, binary=True) -> FiberCamera:
    """A FiberCamera object without a GUI or camera, with FrED's filter switches."""
    cam = FiberCamera.__new__(FiberCamera)
    cam.erode_enabled, cam.dilate_enabled = erode, dilate
    cam.gaussian_enabled, cam.binary_enabled = gaussian, binary
    cam.diameter_coefficient = 1.0  # so the diameter comes out in pixels
    return cam


_default = camera()


def crop(frame: np.ndarray) -> np.ndarray:
    """FrED keeps only the middle half of the rows."""
    h = frame.shape[0]
    return frame[h // 4:3 * h // 4].copy()


def get_edges(frame_rgb: np.ndarray, cam: FiberCamera | None = None):
    return (cam or _default).get_edges(frame_rgb)


def hough(edges: np.ndarray, votes=HOUGH_VOTES, min_length=HOUGH_MIN_LENGTH, max_gap=HOUGH_MAX_GAP):
    return cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, votes,
                           minLineLength=min_length, maxLineGap=max_gap)


def diameter_px(lines) -> float:
    return float(_default.get_fiber_diameter_in_pixels(lines))


def measure(frame_bgr: np.ndarray, cam: FiberCamera | None = None):
    """FrED's camera_loop on one BGR frame -> (diameter_px, lines, edges, binary)."""
    frame = crop(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    edges, binary = get_edges(frame, cam)
    lines = hough(edges)
    return diameter_px(lines), lines, edges, binary


_no_threshold = camera(binary=False)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """FrED's steps before the threshold (crop, gray, erode, dilate, blur), from get_edges."""
    _, smoothed = get_edges(crop(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), _no_threshold)
    return smoothed


def canny(img: np.ndarray, low=100, high=250) -> np.ndarray:
    """The Canny call in FrED's get_edges."""
    return cv2.Canny(img, low, high, apertureSize=3)


def diameter_from_binary(binary: np.ndarray):
    """FrED's steps after the threshold: Canny > Hough > width. Returns (px, lines, edges)."""
    edges = canny(binary)
    lines = hough(edges)
    return diameter_px(lines), lines, edges
