"""Ex 5 - Template matching: where is the fiber?

    python ex5_template.py --index 1

Template matching slides a small image (the template) over the frame and scores how
well it matches at every position. This is the same correlation as Ex 2, except the
kernel is a piece cut from the image itself.

STEPS
  1. Light at 100 %. Press m and drag a box around the wire (include some black on
     both sides), then press Enter. That crop is your template.
  2. WRITE find_template() below (then run  python check.py ex5). A green box marks the
     best match and the bottom bar shows its x position and score.
  3. The wire on FrED cannot move, so move the image instead: the "shift px" slider
     moves the frame sideways in software. Does the detected x follow the shift exactly?
  4. Press b, turn the light down to 30 % (knob, and the slider if you use --synthetic),
     press d. Compare the scores of your method (TM_CCOEFF_NORMED) and of plain TM_CCORR
     (shown in the bottom bar). Which score would you trust for a rule like
     "wire found if score > 0.8"? Why?
  5. Turn the light to 0 %. What does your score say now? How could FrED use this?
"""
import cv2
import numpy as np

import fred
import labkit


def find_template(gray, template):
    """Return (x, y, score) of the best match of `template` in `gray`.

    x, y = top-left corner of the best match; score = its match score.
    Use cv2.matchTemplate with the method cv2.TM_CCOEFF_NORMED, then cv2.minMaxLoc
    on the result. minMaxLoc returns (min_value, max_value, min_location, max_location),
    and each location is an (x, y) tuple. For TM_CCOEFF_NORMED the best match is the MAX.
    """
    # TODO (2 lines)

    raise NotImplementedError("find_template")


labkit.use_answers(globals())


def default_template(gray):
    """A strip around the wire, used until you pick your own with m."""
    columns = np.flatnonzero(gray.mean(axis=0) > (int(gray.min()) + int(gray.max())) / 2)
    if gray.max() - gray.min() < 40 or not columns.size:
        return None
    x0, x1 = max(columns[0] - 30, 0), min(columns[-1] + 30, gray.shape[1])
    y0 = gray.shape[0] // 2 - 40
    return gray[y0:y0 + 80, x0:x1].copy()


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 5 - Template matching", light_slider=True, argv=argv)
    lab.slider("shift px", -150, 150, 0)
    state = {"template": None, "gray": None}

    def pick():
        gray = state["gray"]
        cv2.namedWindow("pick template", cv2.WINDOW_AUTOSIZE)
        x, y, w, h = cv2.selectROI("pick template", gray, showCrosshair=False)
        cv2.destroyWindow("pick template")
        if w > 4 and h > 4:
            state["template"] = gray[y:y + h, x:x + w].copy()
            print(f"template {w} x {h} px taken at x={x}, y={y}")

    lab.action("m", "pick template", pick)

    def process(frame):
        gray = cv2.cvtColor(fred.crop(frame), cv2.COLOR_BGR2GRAY)
        gray = np.roll(gray, lab.value("shift px"), axis=1)
        state["gray"] = gray
        if state["template"] is None:
            state["template"] = default_template(gray)
            if state["template"] is None:
                lab.note("no wire visible: turn the light up to take a template")
                return [("camera", gray), ("template", None)]
        template = state["template"]
        th, tw = template.shape
        shown = labkit.to_bgr(gray)
        panels = [("template (press m to pick a new one)", template, False)]

        found = lab.todo(find_template, gray, template)
        ccorr = cv2.matchTemplate(gray, template, cv2.TM_CCORR)
        _, ccorr_score, _, (cx, _) = cv2.minMaxLoc(ccorr)
        lab.reading("TM_CCORR score", f"{ccorr_score:.3g}")
        lab.reading("TM_CCORR x", cx)
        if found is None:
            return [("camera", shown)] + panels + [("match scores", None)]
        x, y, score = found
        lab.reading("your x px", x)
        lab.reading("your score", f"{score:.3f}")
        lab.reading("wire found (score > 0.8)", "YES" if score > 0.8 else "NO")
        cv2.rectangle(shown, (int(x), int(y)), (int(x) + tw, int(y) + th), (0, 220, 0), 2)
        cv2.line(shown, (int(x) + tw // 2, 0), (int(x) + tw // 2, shown.shape[0]), (0, 220, 0), 1)
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        return [("camera + best match (green)", shown)] + panels + [
            ("TM_CCOEFF_NORMED map (white = 1)", labkit.magnitude(np.maximum(scores, 0), 1.0))]

    lab.run(process)


if __name__ == "__main__":
    main()
