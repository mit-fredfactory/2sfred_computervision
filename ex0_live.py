"""Ex 0 - Warm-up: FrED's diameter measurement, live.

    python ex0_live.py --index 1        (find your index with: python labkit.py --list)

This runs the real FrED code (fiber_camera.py) on your camera. Nothing to write here.

1. Set the light knob to 100 %. Do NOT touch the focus knob.
2. Read the diameter in pixels at the bottom of the window.
3. Turn FrED's filters off and on with the keys e, l (dilate), g, t. For each one:
   press  f  to pick the panel, b  to save it as BEFORE, toggle the filter, then  d
   to see BEFORE | NOW | difference. Which filters change the diameter?
"""
import cv2

import fred
import labkit


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 0 - FrED live", argv=argv)
    lab.toggle("e", "erode", True)
    lab.toggle("l", "dilate", True)
    lab.toggle("g", "gaussian", True)
    lab.toggle("t", "threshold", True)

    def process(frame):
        cam = fred.camera(erode=lab.on("erode"), dilate=lab.on("dilate"),
                          gaussian=lab.on("gaussian"), binary=lab.on("threshold"))
        diameter, lines, edges, binary = fred.measure(frame, cam)
        lab.reading("diameter px", diameter)
        lab.reading("Hough segments", 0 if lines is None else len(lines))
        if diameter == 0:
            lab.note("diameter 0 = FrED found fewer than 2 line segments")
        shown = fred.crop(frame)
        return [("camera (FrED crops to the middle half)", shown),
                ("FrED Hough lines (red)", labkit.draw_lines(shown, lines)),
                ("binary (threshold T=100)", binary),
                ("Canny edges", edges)]

    lab.run(process)


if __name__ == "__main__":
    main()
