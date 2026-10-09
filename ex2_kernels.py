"""Ex 2 - Kernels: Sobel, smoothing, gradient magnitude.

    python ex2_kernels.py --index 1

Your kernels are applied with cv2.filter2D, which slides the
kernel over the image, multiplies, and sums.
Red = positive output, blue = negative output, black = 0.

WRITE (then run  python check.py ex2):
  1. KY    the Sobel kernel for horizontal edges (brightness changing from top to bottom)
  2. BOX   a 3x3 averaging (box) kernel
  3. gradient_magnitude(gx, gy)

PREDICT, THEN LOOK
  a. The wire is vertical. Which panel lights up, Kx or your Ky? Why are the wire's
     left and right edges opposite colors in the Kx panel?
  b. Press r to rotate the image 90 degrees in software (a horizontal wire).
     What happens to Kx and Ky?
  c. Press n to add camera-like noise. Look at Kx. Now press x to blur with BOX first.
     Use b / d to compare Kx before and after the blur. What is the price of blurring?
  d. Press c to flip the kernels by 180 degrees (true convolution instead of correlation).
     What changes in the Kx panel, and what stays the same in the magnitude panel?
  e. Move the "edge threshold" slider. Compare your thresholded magnitude with Canny.
"""
import cv2
import numpy as np

import fred
import labkit

KX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], np.float32)

# TODO 1: Sobel kernel for horizontal edges (3x3).
#         Hint: Kx looks for a change from left to right. Ky looks for a change from top to bottom.
KY = None

# TODO 2: 3x3 box kernel: every pixel becomes the average of its 3x3 neighborhood.
BOX = None


def gradient_magnitude(gx, gy):
    """TODO 3: return the gradient magnitude at every pixel (gx, gy are float arrays)."""
    raise NotImplementedError("gradient_magnitude")


labkit.use_answers(globals())

FULL_SCALE = 4 * 255  # largest possible Sobel output on 8-bit images


def apply_kernel(img, kernel, convolve=False):
    """Slide `kernel` over `img` (float output). convolve=True flips the kernel first."""
    kernel = np.asarray(kernel, np.float32)
    if convolve:
        kernel = cv2.flip(kernel, -1)
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)


def main(argv=None) -> None:
    lab = labkit.Lab("Ex 2 - Kernels", argv=argv)
    lab.toggle("r", "rotate 90")
    lab.toggle("n", "add noise")
    lab.toggle("x", "BOX blur first")
    lab.toggle("c", "flip kernel (convolution)")
    lab.slider("edge threshold", 1, 1020, 300)
    rng = np.random.default_rng(0)

    def process(frame):
        gray = cv2.cvtColor(fred.crop(frame), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if lab.on("rotate 90"):
            gray = np.rot90(gray).copy()
        if lab.on("add noise"):
            gray = np.clip(gray + rng.normal(0, 25, gray.shape), 0, 255).astype(np.float32)
        if lab.on("BOX blur first"):
            if BOX is None:
                lab.note("TODO 2: write BOX to use the blur")
            else:
                gray = apply_kernel(gray, BOX)
        convolve = lab.on("flip kernel (convolution)")

        gx = apply_kernel(gray, KX, convolve)
        panels = [("input", gray), ("Kx: vertical edges", labkit.signed(gx, FULL_SCALE))]
        if KY is None:
            lab.note("TODO 1: write KY")
            return panels + [("your Ky: horizontal edges", None)]
        gy = apply_kernel(gray, KY, convolve)
        panels.append(("your Ky: horizontal edges", labkit.signed(gy, FULL_SCALE)))
        lab.reading("max |gx|", float(np.abs(gx).max()))
        lab.reading("max |gy|", float(np.abs(gy).max()))
        mag = lab.todo(gradient_magnitude, gx, gy)
        if mag is None:
            return panels + [("magnitude", None)]
        threshold = lab.value("edge threshold")
        edges = np.asarray(mag) > threshold
        lab.reading("edge pixels (yours)", f"{int(edges.sum())}")
        canny = fred.canny(np.clip(gray, 0, 255).astype(np.uint8))
        lab.reading("edge pixels (Canny)", f"{int((canny > 0).sum())}")
        return panels + [("your gradient magnitude", labkit.magnitude(mag, FULL_SCALE)),
                         (f"magnitude > {threshold}", edges),
                         ("Canny (FrED's settings)", canny)]

    lab.run(process)


if __name__ == "__main__":
    main()
