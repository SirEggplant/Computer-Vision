import numpy as np
import cv2 as cv
from pathlib import Path
import argparse
from utils import uint8_to_float, text_to_array


def conv2D(image: np.ndarray, h: np.ndarray, **kwargs) -> np.ndarray:
    """Apply a *convolution* operation rather than a *correlation* operation. Using the fact that
    convolution is equivalent to correlation with a flipped kernel, and cv.filter2D implements
    correlation.

    :param image: The input image
    :param h: The kernel
    :param kwargs: Additional arguments to cv.filter2D
    :return: The result of convolving the image with the kernel
    """
    return cv.filter2D(image, -1, cv.flip(h, -1), **kwargs)


def convolution_theorem(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Replicate the behavior of conv2D with borderType=cv.BORDER_CONSTANT, but do it in the
    frequency domain using the convolution theorem.
    """
    img_h, img_w = image.shape[:2]
    filter_h, filter_w = filter.shape

    output_h = img_h + filter_h - 1
    output_w = img_w + filter_w - 1

    F = np.fft.fft2(image, s=(output_h, output_w), axes=(0, 1))

    if image.ndim == 3:
        H = np.fft.fft2(filter, s=(output_h, output_w), axes=(0, 1))
        H = H[:, :, np.newaxis]

    else:
        H = np.fft.fft2(filter, s=(output_h, output_w), axes=(0, 1))

    G = F * H

    g = np.fft.ifft2(G, axes=(0, 1))
    full_result = np.real(g)

    start_h = (filter_h - 1) // 2
    start_w = (filter_w - 1) // 2
    cropped_result = full_result[start_h:start_h + img_h, start_w:start_w + img_w]

    return cropped_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("filter", type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image {args.image} does not exist")
    if not args.filter.exists():
        raise FileNotFoundError(f"Filter {args.filter} does not exist")

    image = uint8_to_float(cv.imread(str(args.image)))
    filter = uint8_to_float(text_to_array(args.filter))

    out1 = conv2D(image, filter, borderType=cv.BORDER_CONSTANT)
    out2 = convolution_theorem(image, filter)

    assert np.allclose(out1, out2, atol=1e-6), "Results do not match"
