"""Ex 3 - Morphology: erosion, dilation, opening.

    python ex3_morphology.py --index 1

Nothing to write; predict first, then check.

The sliders let you change the kernel size and the number of erode / dilate passes.
"width" is the number of white pixels across the wire (median over all rows).

PREDICT, THEN LOOK
  1. Kernel 5, erode 2, dilate 0. How many pixels does the wire lose? (Each pass with a
     5x5 kernel removes 2 px from each side.)  Check your number.
  2. Kernel 5, erode 0, dilate 2. How many pixels does it gain?
  3. Kernel 5, erode 2, dilate 2 (FrED's setting = an "opening"). Width change?
  4. Press p to sprinkle white specks (dust) on the image. Set erode 0, dilate 0, press b,
     then set erode 2, dilate 2 and press d. What happened to the specks? To the wire?
  5. With the specks on, try erode 0 / dilate 2. Why is this a bad idea?
  6. What is the thinnest wire (in px) that FrED's opening can still see?
"""
import cv2
import numpy as np

import fred
import labkit


def wire_width(binary):
    """Median number of white pixels per row (= wire width in px, if there is one wire)."""
    return float(np.median((binary > 0).sum(axis=1)))


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 3 - Morphology", argv=argv)
    lab.slider("kernel size", 1, 15, 5)
    lab.slider("erode passes", 0, 5, 2)
    lab.slider("dilate passes", 0, 5, 2)
    lab.toggle("p", "dust specks")
    specks = {}

    def process(frame):
        gray = cv2.cvtColor(fred.crop(frame), cv2.COLOR_BGR2GRAY)
        if lab.on("dust specks"):
            if gray.shape not in specks:  # same specks every frame, so before/after compare
                rng = np.random.default_rng(1)
                mask = np.zeros(gray.shape, np.uint8)
                for _ in range(60):
                    x, y = int(rng.integers(0, gray.shape[1])), int(rng.integers(0, gray.shape[0]))
                    cv2.circle(mask, (x, y), int(rng.integers(1, 4)), 255, -1)
                specks[gray.shape] = mask
            gray = np.maximum(gray, specks[gray.shape])
        _, binary = cv2.threshold(gray, fred.FRED_T, 255, cv2.THRESH_BINARY)

        size = lab.value("kernel size")
        kernel = np.ones((size, size), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=lab.value("erode passes"))
        result = cv2.dilate(eroded, kernel, iterations=lab.value("dilate passes"))

        before_w, after_w = wire_width(binary), wire_width(result)
        diameter, lines, _ = fred.diameter_from_binary(result)
        lab.reading("width before px", before_w)
        lab.reading("width after px", after_w)
        lab.reading("change px", after_w - before_w)
        lab.reading("FrED diameter px", diameter)
        n_before = cv2.connectedComponents(binary)[0] - 1
        n_after = cv2.connectedComponents(result)[0] - 1
        lab.reading("white blobs", f"{n_before} -> {n_after}")
        return [("binary (T=100), no morphology", binary),
                (f"after erode x{lab.value('erode passes')}", eroded),
                (f"after erode + dilate x{lab.value('dilate passes')}",
                 labkit.draw_lines(result, lines))]

    lab.run(process)


if __name__ == "__main__":
    main()
