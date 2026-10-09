"""Ex 1 - An image is a grid of numbers; Otsu's threshold vs the light knob.

    python ex1_otsu.py --index 1

PART A (look)
  Press space to freeze. The "patch" panel shows the actual numbers of a 7 x 11 patch
  across the wire's left edge. How many pixels wide is the dark-to-bright transition?

PART B (write code): fill in otsu_threshold() below, then run  python check.py ex1

PART C (light knob experiment)
  For each light setting 100 %, 50 %, 20 %, 0 %:
    - you can adjust the light intensity using the switch on the USB cable, set the "light %" slider in the GUI to the same number (it labels your snapshot)
    - wait 2 s, press  s  to save a snapshot
  Then compare them:   python compare.py --latest 4
  Questions:
    1. How does the histogram change as the light goes down?
    2. What is the limitation if we fix T = 100?
    3. At 0 % Otsu still returns a threshold. Is that result meaningful? (Hint: Otsu splits
       the histogram into TWO classes. How many classes are really there at 0 %?)
"""
import cv2
from matplotlib.pyplot import gray
import numpy as np

import fred
import labkit

FRED_T = 100  # fiber_camera.py get_edges: cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)


def otsu_threshold(gray):
    """Return (k, binary): Otsu's threshold k and the 0/255 binary image.

    FrED uses a fixed threshold:
        k, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    Change it so OpenCV picks k with Otsu's method (add the flag cv2.THRESH_OTSU;
    the 100 is then ignored, so you can pass 0).
    """
    # TODO (1 line)
    
    # return
    raise NotImplementedError("otsu_threshold")


labkit.use_answers(globals())


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 1 - Otsu vs light", light_slider=True, argv=argv)

    def process(frame):
        if not lab.source.synthetic:
            lab.note("light % slider: set it to the knob on FrED (it labels your snapshots)")
        gray = fred.preprocess(frame)  # FrED's crop, gray, erode, dilate, blur
        fixed_k, fixed = cv2.threshold(gray, FRED_T, 255, cv2.THRESH_BINARY)
        fixed_px, fixed_lines, _ = fred.diameter_from_binary(fixed)
        lab.reading("fixed T", fixed_k)
        lab.reading("diameter fixed T px", fixed_px)

        marks = [(FRED_T, (0, 0, 255), "fixed T")]
        result = lab.todo(otsu_threshold, gray)
        otsu_panel = None
        if result is not None:
            otsu_k, otsu = result
            otsu_px, otsu_lines, _ = fred.diameter_from_binary(otsu)
            lab.reading("Otsu k*", otsu_k)
            lab.reading("diameter Otsu px", otsu_px)
            marks.append((otsu_k, (0, 220, 0), "Otsu k*"))
            otsu_panel = labkit.draw_lines(otsu, otsu_lines, (0, 220, 0))

        row = gray.shape[0] // 2
        cols = np.flatnonzero(gray[row] > (int(gray[row].min()) + int(gray[row].max())) // 2)
        x0 = int(cols[0]) - 5 if cols.size else gray.shape[1] // 2
        boxed = labkit.to_bgr(gray)
        cv2.rectangle(boxed, (x0 - 1, row - 4), (x0 + 11, row + 4), (0, 220, 255), 1)
        lab.reading("gray.shape", str(gray.shape))
        lab.reading("dtype", str(gray.dtype))
        return [("gray (yellow box = patch)", boxed),
                (f"patch: rows {row - 3}-{row + 3}, cols {x0}-{x0 + 10}",
                 labkit.patch_view(gray, x0, row - 3), False),
                ("histogram (log counts)", labkit.histogram(gray, marks), False),
                ("FrED fixed T=100", labkit.draw_lines(fixed, fixed_lines)),
                ("your Otsu threshold", otsu_panel)]

    lab.run(process)


if __name__ == "__main__":
    main()
