import cv2 as cv
import numpy as np


def my_harris_corners(image: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """Given a grayscale image in uint8 format, compute the Harris response function
        $$R = λ1*λ2 − k*(λ1 + λ2)^2$$
    at each pixel.

    This function must replicate the behavior or cv.cornerHarris (assume ksize=3), but you may not
    call that function. You *may* use cv.Sobel, cv.boxFilter, and other 'low-level' operations, and
    you may use any numpy functions you like. The returned array will contain floating point values.
    It is the responsibility of the caller to handle normalization, non-maximal suppression, and
    thresholding (so don't worry about those here). Since the caller will normalize, the output of
    this function may equal some *scaled* version of the output of cv.cornerHarris.

    :param image: A grayscale image in uint8 format
    :param block_size: The size of the neighborhood to consider for each pixel when computing the
        second moment matrix
    :param k: A constant used to tune the sensitivity of the corner detector
    :return: The Harris response function R at each pixel, as a floating point array of the same
        shape as the input image
    """
    # Get gradient
    gx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=3)
    # Sum using box filter
    gx2sum = cv.boxFilter(gx * gx, cv.CV_32F, (block_size, block_size))
    gy2sum = cv.boxFilter(gy * gy, cv.CV_32F, (block_size, block_size))
    gxgysum = cv.boxFilter(gx * gy, cv.CV_32F, (block_size, block_size))
    # Compute Harris Corners
    det = (gx2sum * gy2sum) - (gxgysum * gxgysum)
    tr = gx2sum + gy2sum
    return det - k * (tr * tr)


def locate_corners(
    harris: np.ndarray,
    threshold: float,
    nms_block_size: int = 5,
) -> list[tuple[int, int]]:
    """Given a Harris response function, return the (i, j) coordinates of identified corners.

    :param harris: The Harris response function, i.e. the normalized output of my_harris_corners()
        or cv.cornerHarris()
    :param threshold: The threshold above which a corner is considered to be present
    :param nms_block_size: The size of the neighborhood to consider when performing non-maximal
        suppression
    :return: An array of (x, y) coordinates of identified corners
    """
    # Apply threshold
    candidate = harris > threshold
    # Non-Maximum Suppression
    corners = []
    # Go through all pixels in the image
    for i in range(harris.shape[0]):
        for j in range(harris.shape[1]):
            # Only process pixels that passes threshold
            if not candidate[i, j]:
                continue
            # Define neightborhood boundaries
            istart = max(0, i - (nms_block_size // 2))
            iend = min(harris.shape[0], i + (nms_block_size // 2) + 1)
            jstart = max(0, j - (nms_block_size // 2))
            jend = min(harris.shape[1], j + (nms_block_size // 2) + 1)
            # Extract neighbors
            neighbors = harris[istart:iend, jstart:jend]
            # Find the maximum value neighbor
            if harris[i, j] == np.max(neighbors):
                corners.append((j, i))

    return corners


def main(image_filename: str, blur_size=5, block_size=5, k=0.04):
    image = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    blurred = cv.GaussianBlur(image, (blur_size, blur_size), 0)

    opencv_harris = cv.cornerHarris(
        blurred,
        blockSize=block_size,
        ksize=3,
        k=k,
    )
    my_harris = my_harris_corners(blurred, block_size, k)

    # Normalize both so we don't have to worry about scale
    opencv_harris = opencv_harris / opencv_harris.max()
    my_harris = my_harris / my_harris.max()

    # Create a window with a slider for the threshold
    window_name = "Harris Corners (OpenCV left / Custom right)"
    cv.namedWindow(window_name)

    def draw(threshold):
        opencv_xy = locate_corners(opencv_harris, threshold)
        my_xy = locate_corners(my_harris, threshold)

        opencv_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        my_display = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        for x, y in opencv_xy:
            cv.circle(opencv_display, (x, y), 3, (0, 0, 255), -1)
        for x, y in my_xy:
            cv.circle(my_display, (x, y), 3, (0, 0, 255), -1)

        cv.imshow(window_name, np.hstack([opencv_display, my_display]))

    def on_slider_update(slider_value):
        threshold = slider_value / 100
        draw(threshold)

    cv.createTrackbar("Threshold", window_name, 0, 100, on_slider_update)
    cv.setTrackbarPos("Threshold", window_name, 50)

    while cv.waitKey(1) != ord("q"):
        pass
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="The image to process")
    parser.add_argument(
        "--blur-size", type=int, default=5, help="The size of the Gaussian blur kernel"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="The size of the neighborhood to consider for each pixel",
    )
    parser.add_argument(
        "-k",
        type=float,
        default=0.04,
        help="A constant used to tune the sensitivity of the corner detector",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error(f"The file {args.image} does not exist")
    main(args.image, args.blur_size, args.block_size, args.k)
