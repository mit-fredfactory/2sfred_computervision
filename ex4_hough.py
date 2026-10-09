"""Ex 4 - From edges to lines to one number: Canny, Hough, diameter.

    python ex4_hough.py --index 1

The Hough transform does not find edges itself: every white pixel of its input votes for
the lines through it. Canny turns the solid white wire into two thin boundary lines,
so Hough finds exactly the two sides of the wire.

PREDICT, THEN LOOK
  1. "min line length": raise it from 30 toward 300. When do the red segments disappear,
     and what does the diameter read then? (FrED needs at least 2 segments.)
  2. "votes": same question. What does one "vote" correspond to?
  3. "max line gap": press p to add dust gaps along the wire edges, then lower the gap
     from 100 to 0. Why does FrED use a large gap?
  4. Canny thresholds: move "canny low" / "canny high". Does FrED's diameter change?
     Press g to feed Canny the GRAY image instead of the binary one and try again.
     Why do the thresholds do nothing on a binary image? (Slide 64.)

WRITE (then run  python check.py ex4)
  boundary(binary): another way to get the wire outline without Canny (see below).
  Does the diameter from your boundary match the one from Canny?
"""
import cv2
import numpy as np

import fred
import labkit


def boundary(binary):
    """TODO: return the 1-px outline of the white regions of a 0/255 binary image.

    Use the "morphological gradient": the binary image minus its erosion with a
    3x3 kernel of ones (np.ones((3, 3), np.uint8)). Only pixels on the border of a
    white region survive.
    """
    # TODO (1 line)
    raise NotImplementedError("boundary")


labkit.use_answers(globals())


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 4 - Canny and Hough", argv=argv)
    lab.slider("min line length", 0, 300, fred.HOUGH_MIN_LENGTH)
    lab.slider("max line gap", 0, 200, fred.HOUGH_MAX_GAP)
    lab.slider("votes", 1, 200, fred.HOUGH_VOTES)
    lab.slider("canny low", 1, 1000, 100)
    lab.slider("canny high", 1, 1000, 250)
    lab.toggle("p", "dust gaps")
    lab.toggle("g", "Canny on gray")

    def process(frame):
        smoothed = fred.preprocess(frame)
        if lab.on("dust gaps"):  # black specks breaking the wire's edges every ~40 rows
            smoothed = smoothed.copy()
            columns = np.flatnonzero(smoothed.mean(axis=0) > 100)
            if columns.size:
                for y in range(10, smoothed.shape[0], 40):
                    for x in (columns[0], columns[-1]):
                        cv2.circle(smoothed, (int(x), y), 8, 0, -1)
        _, binary = cv2.threshold(smoothed, fred.FRED_T, 255, cv2.THRESH_BINARY)
        source = smoothed if lab.on("Canny on gray") else binary
        edges = fred.canny(source, lab.value("canny low"), lab.value("canny high"))

        def hough(img):
            return fred.hough(img, lab.value("votes"), lab.value("min line length"),
                              lab.value("max line gap"))

        lines = hough(edges)
        lab.reading("Canny edge pixels", f"{int((edges > 0).sum())}")
        lab.reading("segments", 0 if lines is None else len(lines))
        lab.reading("FrED diameter px", fred.diameter_px(lines))
        shown = fred.crop(frame)
        panels = [(f"binary{' (with dust gaps)' if lab.on('dust gaps') else ''}", binary),
                  (f"Canny on {'GRAY' if lab.on('Canny on gray') else 'binary'}", edges),
                  ("Hough segments (red)", labkit.draw_lines(shown, lines))]

        outline = lab.todo(boundary, binary)
        if outline is None:
            return panels + [("your boundary()", None)]
        your_lines = hough(np.asarray(outline, np.uint8))
        lab.reading("diameter from your boundary px", fred.diameter_px(your_lines))
        return panels + [("your boundary()", outline),
                         ("Hough on your boundary (green)",
                          labkit.draw_lines(shown, your_lines, (0, 220, 0)))]

    lab.run(process)


if __name__ == "__main__":
    main()
