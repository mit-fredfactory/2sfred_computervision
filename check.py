"""Check your answers.

    python check.py          check everything
    python check.py ex2      check one exercise

Every check prints PASS or FAIL with a hint. No camera needed.
"""
from __future__ import annotations

import importlib
import sys
import traceback

import cv2
import numpy as np

import labkit

RESULTS = []


def report(name: str, ok: bool, hint: str = "") -> None:
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        hint: {hint}"))


def attempt(name: str, fn, hint: str):
    """Run fn(); a missing TODO or a crash counts as FAIL."""
    try:
        return fn()
    except NotImplementedError:
        report(name, False, "not written yet (still raises NotImplementedError)")
    except Exception as error:  # show students what broke
        report(name, False, f"{type(error).__name__}: {error}. {hint}")
    return None


def gray_wire(light=100, seed=0):
    frame = labkit.synthetic_frame(light, rng=np.random.default_rng(seed))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def check_ex1(m):
    print("ex1 - Otsu")
    for light in (100, 30):
        gray = gray_wire(light)
        result = attempt(f"otsu_threshold at light {light} %", lambda: m.otsu_threshold(gray),
                         "return the two values that cv2.threshold returns")
        if result is None:
            return
        want_k, want = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k, binary = result
        report(f"otsu_threshold at light {light} %: k = {float(k):.0f}",
               abs(float(k) - want_k) < 1 and np.array_equal(binary, want),
               f"expected k = {want_k:.0f}. Did you add cv2.THRESH_OTSU to cv2.THRESH_BINARY?")


def check_ex2(m):
    print("ex2 - kernels")
    want_ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    if m.KY is None:
        report("KY", False, "KY is still None")
    else:
        ky = np.asarray(m.KY, np.float32)
        ok = ky.shape == (3, 3) and np.array_equal(ky, want_ky)
        flipped = ky.shape == (3, 3) and np.array_equal(ky, -want_ky)
        report("KY is the Sobel kernel for horizontal edges", ok,
               "you have the right kernel with the opposite sign; use the convention where "
               "a dark-above / bright-below edge is positive (like Kx: dark-left / bright-right)"
               if flipped else "Ky should be Kx turned on its side: the weights 1, 2, 1 go "
               "along a row, with -1 -2 -1 on top and 1 2 1 at the bottom")
        if ok:  # the kernel should answer to horizontal edges only
            img = np.zeros((20, 20), np.float32)
            img[10:] = 255
            gy = cv2.filter2D(img, cv2.CV_32F, ky)
            report("KY responds to a horizontal edge (+1020) and not to a vertical one",
                   gy.max() == 1020 and np.abs(cv2.filter2D(img.T.copy(), cv2.CV_32F, ky)).max() == 0)
    if m.BOX is None:
        report("BOX", False, "BOX is still None")
    else:
        box = np.asarray(m.BOX, np.float32)
        report("BOX is a 3x3 averaging kernel",
               box.shape == (3, 3) and np.allclose(box, 1 / 9),
               "a box kernel has 9 equal weights that add up to 1"
               + (" (yours add up to %.3g)" % box.sum() if box.shape == (3, 3) else ""))
    gx = np.array([[3.0, -6.0], [0.0, 5.0]])
    gy = np.array([[4.0, 8.0], [0.0, -12.0]])
    mag = attempt("gradient_magnitude", lambda: m.gradient_magnitude(gx, gy), "use np.sqrt")
    if mag is not None:
        report("gradient_magnitude", np.allclose(mag, [[5, 10], [0, 13]]),
               "magnitude = sqrt(gx^2 + gy^2); it is never negative. "
               "(|gx| + |gy| is an approximation, not the exact value)")


def check_ex4(m):
    print("ex4 - boundary")
    binary = np.zeros((40, 60), np.uint8)
    binary[:, 20:35] = 255
    outline = attempt("boundary", lambda: m.boundary(binary), "binary - cv2.erode(binary, kernel)")
    if outline is None:
        return
    outline = np.asarray(outline)
    want = binary - cv2.erode(binary, np.ones((3, 3), np.uint8))
    report("boundary of a 15 px wide bar", np.array_equal(outline, want),
           "keep only the white pixels that the erosion removes: binary minus its erosion")
    report("boundary is uint8 0/255 (not bool, not negative)",
           outline.dtype == np.uint8 and set(np.unique(outline)) <= {0, 255},
           "with uint8 images, binary - eroded is never negative because eroded <= binary")


def check_ex5(m):
    print("ex5 - template matching")
    gray = gray_wire()
    template = gray[200:280, 270:490].copy()
    for shift in (0, 37):
        moved = np.roll(gray, shift, axis=1)
        found = attempt(f"find_template, wire shifted {shift} px",
                        lambda: m.find_template(moved, template),
                        "return x, y, score from cv2.minMaxLoc")
        if found is None:
            return
        x, y, score = found
        report(f"find_template, wire shifted {shift} px: x = {x}",
               abs(x - (270 + shift)) <= 1 and abs(y - 200) <= 3 and score > 0.95,
               "use TM_CCOEFF_NORMED and the MAX location of cv2.minMaxLoc; "
               "locations are (x, y), x first")
    dark = gray_wire(30)
    found = attempt("find_template at 30 % light", lambda: m.find_template(dark, template), "")
    if found is not None:
        report("score stays high when the light drops (normalized method)", found[2] > 0.9,
               "use cv2.TM_CCOEFF_NORMED (not TM_CCORR or TM_SQDIFF)")


def check_ex6(m):
    print("ex6 - calibration")
    c = attempt("calibration_coefficient", lambda: m.calibration_coefficient(150.0),
                "mm_per_px = KNOWN_MM / average_px")
    if c is not None:
        report(f"calibration_coefficient(150 px) = {c:.5f}", np.isclose(c, m.KNOWN_MM / 150),
               "mm per pixel = known diameter in mm / average width in pixels")
    mm = attempt("px_to_mm", lambda: m.px_to_mm(150.0, 0.01), "")
    if mm is not None:
        report("px_to_mm(150 px, 0.01 mm/px) = 1.5 mm", np.isclose(mm, 1.5),
               "pixels times mm per pixel")


CHECKS = {"ex1": ("ex1_otsu", check_ex1), "ex2": ("ex2_kernels", check_ex2),
          "ex4": ("ex4_hough", check_ex4), "ex5": ("ex5_template", check_ex5),
          "ex6": ("ex6_calibration", check_ex6)}


def main() -> int:
    wanted = sys.argv[1:] or list(CHECKS)
    for name in wanted:
        if name not in CHECKS:
            print(f"unknown exercise {name!r}; choose from {', '.join(CHECKS)}")
            return 2
        module_name, check = CHECKS[name]
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print(f"{name}: could not import {module_name}.py:")
            traceback.print_exc()
            RESULTS.append(False)
            continue
        check(module)
    passed = sum(RESULTS)
    print(f"\n{passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
