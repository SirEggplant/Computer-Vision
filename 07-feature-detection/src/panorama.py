import cv2 as cv
import numpy as np
from utils import float32_to_uint8, uint8_to_float32
from pathlib import Path
from typing import Optional


def add_weights_channel(image: np.ndarray, border_size: int = 0) -> np.ndarray:
    """Given an image, add a 4th channel to it that contains weights for blending. The high-level
    idea is that the weights near the borders of the image should be low to reduce visible 'seams'
    where the edge of one image lies near the middle of another.
    """
    h, w = image.shape[:2]
    # Let's say an image is size 200x100. Then, we can't have a 'border' of size bigger than 50
    # because then the size of top+bottom borders would be greater than the height of the image.
    # This line makes sure that the border size is at most half the height and half the width.
    border_size = min(border_size, h // 2, w // 2)

    # The weights_x and weights_y arrays are 1D arrays of size w and h, respectively. They ramp
    # up from 0 to 1 in 'border_size' steps, then stay at 1 for the middle section of the image,
    # then ramp back down to 0 in 'border_size' steps.
    # Graphically, the weights look like this:
    #
    #    /-----------------\
    #   /                   \
    #  /                     \
    #  <-> size of this 'ramp' is 'border_size'
    weights_x = np.hstack(
        [
            np.linspace(0, 1, border_size),
            np.ones(w - 2 * border_size),
            np.linspace(1, 0, border_size),
        ]
    )
    weights_y = np.hstack(
        [
            np.linspace(0, 1, border_size),
            np.ones(h - 2 * border_size),
            np.linspace(1, 0, border_size),
        ]
    )
    weights_xy = np.minimum(weights_y[:, np.newaxis], weights_x[np.newaxis, :]).astype(np.float32)
    return np.dstack((image, weights_xy))


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Given a 3x3 homography matrix and an 2xn array of 2D points (each point as a column vector),
    return the new 2xn array of points after applying the homography.
    """
    _, n = points.shape
    points_augmented = np.vstack((points, np.ones((1, n))))
    # Matrix multiplication of h with points_augmented. The '@' operator is a built-in python
    # operator which numpy interprets as 'matrix multiplication'.
    new_points = h @ points_augmented
    new_points = new_points[:2] / new_points[2]
    return new_points


def homography_then_translate(homog: np.ndarray, translate: tuple[float, float]) -> np.ndarray:
    """Given a 3x3 homography matrix 'homog' that will map from coordinates x to coordinates y like
    y = homog @ x, return a new homography matrix homog2 such that y2 = homog2 @ x is equivalent to
    y2 = homog @ x + t for the given translation vector t. In other words, this function combines
    "homography followed by translation" into a single homography matrix.
    """
    # Make 'translator' - another 3x3 homography matrix that *just* does translation
    translator = np.eye(3)
    translator[:2, 2] = np.array(translate)
    # The idea is that translator @ (homog @ x) is the same as (homog @ x) + translate, and we can
    # group the parentheses differently to get (translator @ homog) @ x
    return translator @ homog


def calculate_bounding_box_of_warps(homographies, images):
    """Given a list of homography matrices and a list of images, compute the bounding box of the
    warped images by seeing where their corners land after warping.

    To do this, we'll compute the new 2D location of the *corners* of each image. Then, we'll
    compute the bounding box of all of those points.

    Returns the (left, top, width, height) of the bounding box.
    """
    warped_corners = np.hstack(
        [
            apply_homography(
                h,
                np.array(
                    [[0, 0], [im.shape[1], 0], [0, im.shape[0]], [im.shape[1], im.shape[0]]]
                ).T,
            )
            for h, im in zip(homographies, images)
        ]
    )
    # Compute the bounding box that contains all the warped corners
    left = int(np.floor(np.min(warped_corners[0])))
    right = int(np.ceil(np.max(warped_corners[0])))
    top = int(np.floor(np.min(warped_corners[1])))
    bottom = int(np.ceil(np.max(warped_corners[1])))
    w, h = right - left, bottom - top
    return left, top, w, h


def weighted_blend(images: list[np.ndarray]) -> np.ndarray:
    """Given a list of images each with 4 channels (e.g. BGRW or LABW), blend them together using
    the weights in the 4th channel. Return the blended image. In the formulas below, image[i] refers
    to just the first three channels of the i'th image (i.e. BGR or LAB or whatever), and w[i]
    refers to the value in the 4th 'weight' channel of the image.

    Each output pixel will be a weighted average of the two corresponding input pixels, so

        output = sum(w[i] * image[i]) / sum(w)

    Anywhere that sum(w) is zero, the outputs will be zero.
    """
    weights = [im[:, :, 3:] for im in images]
    images = [im[:, :, :3] for im in images]
    sum_of_weights = np.sum(weights, axis=0)
    sum_of_weights[sum_of_weights == 0] = 1  # Avoid divide by zero
    return np.sum([w * im for w, im in zip(weights, images)], axis=0) / sum_of_weights


def my_find_homography(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Given two sets of points (nx2 arrays), compute the 3x3 homography that maps the first to
    the second.
    """
    n = points1.shape[0]
    assert n == points2.shape[0] and n >= 4, "Both sets of points must have at least 4 points"
    # Build the design matrix
    design_matrix = np.zeros((2 * n, 8))
    b = np.zeros((2 * n,))
    for i in range(n):
        x, y = points1[i]
        u, v = points2[i]
        design_matrix[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        design_matrix[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v

    # Solve for the homography vector 'h' that minimizes the error in the least squares sense
    h, residuals, rank, s = np.linalg.lstsq(design_matrix, b, rcond=None)

    # Reshape 'h' to a 3x3 matrix and set the bottom-right element to 1
    homography_matrix = np.append(h, 1).reshape((3, 3))

    return homography_matrix


def find_homography_ransac(
    points_src: np.ndarray, points_dst: np.ndarray, n_iters: int = 1000, threshold: float = 3.0
) -> np.ndarray:
    """Given two sets of points (nx2 arrays), compute the 3x3 homography that maps the first
    (source) coordinates to the second (destination) coordinates. This function should use RANSAC to
    robustly handle outliers. The 'n_iters' parameter is the number of RANSAC iterations to perform.
    See the README.pdf instructions and online resources for more details.

    You may not use any OpenCV functions here. You should use the my_find_homography function
    above to compute the homography for each randomly sampled subset (rather than calling
    cv.findHomography()).
    """
    bestinliers = None
    bestH = None
    maxinliers = 0
    npoints = points_src.shape[0]

    for _ in range(n_iters):
        # Randomly sample 4 point pairs
        indices = np.random.choice(npoints, size=4, replace=False)
        # Compute homography from these points
        H = my_find_homography(points_src[indices], points_dst[indices])
        # Transform all source points using this homography
        transformed = apply_homography(H, points_src.T)
        # Compute distances between transformed and actual points
        errors = np.linalg.norm(transformed - points_dst.T, axis=0)
        # Find inliers
        ninliers = np.sum(errors < threshold)
        # Update best model
        if ninliers > maxinliers:
            maxinliers = ninliers
            bestinliers = errors < threshold
            bestH = H

    # Find good sets of inliers
    if maxinliers >= 4:
        bestH = my_find_homography(points_src[bestinliers], points_dst[bestinliers])

    return bestH if bestH is not None else np.eye(3)


def get_matching_points(
    image_ref: np.ndarray, image_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given a pair of images, return a tuple of (reference points, query points), where each set
    of points is an (n, 2) sized array of keypoint locations such that points_r[i] is the closest
    match (according to descriptors, disregarding spatial location) to points_q[i]. Returns one
    match for each keypoint in the query image (its closest match in the reference image).

    For more information, see the OpenCV tutorial here:
    https://opencv2-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    """
    sift = cv.SIFT.create()
    kp1, des1 = sift.detectAndCompute(image_ref, None)
    kp2, des2 = sift.detectAndCompute(image_query, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("No descriptors found in one of the images")
        return np.array([]), np.array([])

    # Use BFMatcher with Lowe's ratio test
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    goodmatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodmatches.append(m)

    print(f"Good matches (Lowe): {len(goodmatches)}")

    if len(goodmatches) < 4:
        return np.array([]), np.array([])

    query_pts = np.array([kp2[m.queryIdx].pt for m in goodmatches])
    ref_pts = np.array([kp1[m.trainIdx].pt for m in goodmatches])

    return ref_pts, query_pts


def stitch(
    images: list[np.ndarray],
    reference_image_index: int,
    border_blend_size: int,
) -> np.ndarray:
    """This function puts all the pieces together and returns a composited panorama image. The
    'images' list contains the images to stitch together. The 'reference_image_index' is the
    index of the image that will be used as the reference coordinate system for the output
    panorama. The 'border_blend_size' is the size of the border blending region in pixels.
    """
    # Get matching points between each image and the reference image. point_pairs[i] will contain
    # (reference_points, image_points) for the i'th image compared to the reference image.
    point_pairs = [get_matching_points(images[reference_image_index], im) for im in images]

    # Compute 3x3 homography matrix that maps from images[i]'s coordinate system to image[ref]'s
    # coordinate system where image[ref] is the reference image and images[i] is any other image.
    homographies = [find_homography_ransac(pts_i, pts_ref) for pts_ref, pts_i in point_pairs]

    # Convert images to floating point representation so that we don't have to worry about overflow
    # or underflow while doing math.
    images = [uint8_to_float32(im) for im in images]

    # Create the 'weights' for each pixel in each image as a 4th channel.
    images = [add_weights_channel(im, border_blend_size) for im in images]

    # Next, we need to calculate how big of an output image we need to hold all the warped images.
    left, top, new_w, new_h = calculate_bounding_box_of_warps(homographies, images)

    # We can't use negative rows/cols to index into images. If 'left' or 'top' is negative, then
    # we need to adjust all the homography matrices so that they add back in a translation of
    # (|left|, |top|). This will shift all the images to the right and/or down so that the top-left
    # corner of the combined image is at (0, 0).
    x_shift, y_shift = max(0, -left), max(0, -top)
    homographies = [homography_then_translate(h, (x_shift, y_shift)) for h in homographies]

    # Warp all images (plus their 4th 'weights' channel) into the new coordinate system
    warped = [cv.warpPerspective(im, h, (new_w, new_h)) for h, im in zip(homographies, images)]

    # Blend the warped images together using the weights in the 4th channel (note that the weights
    # were also transformed by the homography)
    stitched = weighted_blend(warped)

    return float32_to_uint8(stitched)


def main(
    images: list[np.ndarray], reference_index: int, border_size: int, output: Optional[Path] = None
):
    panorama = stitch(images, reference_index, border_size)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(output), panorama)
    else:
        cv.imshow("Panorama", panorama)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+", help="Space-separated paths to the images")
    parser.add_argument(
        "-b", "--border-size", type=int, default=50, help="Size of the border blend"
    )
    parser.add_argument(
        "-r", "--reference-index", type=int, default=0, help="Index of the reference image"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Path to save the combined image"
    )
    args = parser.parse_args()

    for i, image in enumerate(args.images):
        if not image.exists():
            raise FileNotFoundError(f"Image {i + 1} does not exist: {image}")
        else:
            args.images[i] = cv.imread(str(image))

    main(**vars(args))
