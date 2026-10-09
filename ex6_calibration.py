"""Ex 6 - Calibration: from pixels to millimeters.

    python ex6_calibration.py --index 1

The camera measures the wire in pixels. To get millimeters, FrED looks
at a wire of KNOWN diameter, averages its pixel width over 50 frames and stores
    mm_per_px = known diameter / average pixel width
(FrED calls this diameter_coefficient). After that, every reading is px * mm_per_px.

WRITE (then run  python check.py ex6)
  1. calibration_coefficient(average_px)
  2. px_to_mm(px, mm_per_px)

STEPS
  1. Light at 100 %. Press c to collect 50 samples (about 2 s). The bottom bar shows the
     coefficient and the live diameter in mm. Is it close to KNOWN_MM? How much is 1 px in um?
"""
import cv2
import numpy as np

import fred
import labkit

KNOWN_MM = 1.5   # outer diameter of our 22 AWG wire (with insulation), measured with calipers
NUM_SAMPLES = 50  # fiber_camera.py calibrate(): num_samples = 50


def calibration_coefficient(average_px):
    """TODO 1: mm per pixel, from the average pixel width of the KNOWN_MM wire (1 line)."""
    raise NotImplementedError("calibration_coefficient")


def px_to_mm(px, mm_per_px):
    """TODO 2: convert a width in pixels to millimeters (1 line)."""
    raise NotImplementedError("px_to_mm")


labkit.use_answers(globals())


def calibrate(samples):
    """FrED's calibrate() averaging loop."""
    accumulated, n = 0.0, 0
    for w in samples:
        if w > 0:  # get_fiber_diameter_in_pixels() returns 0 when it finds no wire
            accumulated += w
            n += 1
    average = accumulated / n if n > 0 else 1  # FrED: "Prevent division by zero"
    return average, n


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 6 - Calibration", argv=argv)
    state = {"samples": [], "collecting": False, "mm_per_px": None}

    def start():
        state["samples"], state["collecting"] = [], True
        print(f"collecting {NUM_SAMPLES} samples...")

    lab.action("c", "collect 50 samples", start)

    def finish():
        samples = state["samples"]
        average, n = calibrate(samples)
        print(f"samples (px): {[round(s, 1) for s in samples]}")
        print(f"{n} samples used, average {average:.2f} px")
        coefficient = lab.todo(calibration_coefficient, average)
        if coefficient is not None:
            state["mm_per_px"] = coefficient
            print(f"mm_per_px: {coefficient:.5f}  ({coefficient * 1000:.1f} um per pixel)")

    def process(frame):
        px, lines, _, _ = fred.measure(frame)
        if state["collecting"]:
            state["samples"].append(px)
            if len(state["samples"]) >= NUM_SAMPLES:
                state["collecting"] = False
                finish()
        samples = state["samples"]
        lab.reading("diameter px", px)
        if state["collecting"]:
            lab.reading("collecting", f"{len(samples)}/{NUM_SAMPLES}")
        elif samples:
            average, n = calibrate(samples)
            lab.reading("average px", f"{average:.1f} ({n} used)")
        coefficient = state["mm_per_px"]
        if coefficient is None:
            if not state["collecting"]:
                lab.note("press c to calibrate")
        else:
            lab.reading("mm_per_px", f"{coefficient:.5f}")
            mm = lab.todo(px_to_mm, px, coefficient)
            if mm is not None:
                lab.reading("diameter mm", f"{mm:.3f}")
        plot = labkit.line_plot(samples, y_max=max(samples + [px, 1]) * 1.2)
        labkit.put_text(plot, f"calibration samples (px), {len(samples)}/{NUM_SAMPLES}", (40, 16))
        return [("camera + FrED lines", labkit.draw_lines(fred.crop(frame), lines)),
                ("samples", plot, False)]

    lab.run(process)


if __name__ == "__main__":
    main()
