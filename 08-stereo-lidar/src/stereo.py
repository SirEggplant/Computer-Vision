from pathlib import Path

import cv2 as cv
import numpy as np
import open3d as o3d

from utils import ceil_16, fix16_to_float32, plot_3d_points


def disparity_to_3d(
    disparities: np.ndarray,
    focal_length: float,
    baseline_mm: float,
    dmin: float = 0.0,
) -> np.ndarray:
    """
    Convert a disparity map to 3D coordinates.

    Disparity formula is Z = f * B / (d + dmin), where:
    - Z is the true depth of the pixel in mm,
    - f is the focal length of the camera in pixels,
    - B is the baseline distance between cameras in mm,
    - d is the disparity value in pixels,
    - dmin is a value added to each disparity value (e.g. to account for image cropping).

    Any values where d=0 should be treated as missing/ignored.

    Args:
        disparities (np.ndarray): (h, w, 1) disparity map.
        focal_length (float): Focal length of the camera in units of pixels.
        baseline_mm (float): Baseline x distance between cameras in millimeters.
        dmin (float): value added to each disparity value (e.g. to account for image cropping).

    Returns:
        - (N, 3) array of 3D points in (x, y, z) format, all in mm units, where N is the number of
            pixels with nonzero disparity values.
    """
    h, w = disparities.shape[:2]

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    valid_mask = disparities > 0
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    disparities_valid = disparities[valid_mask]

    disparities_valid = disparities_valid + dmin

    Z = (focal_length * baseline_mm) / disparities_valid

    center_x = w / 2
    center_y = h / 2

    X = (x_coords - center_x) * Z / focal_length
    Y = (y_coords - center_y) * Z / focal_length

    points_3d = np.column_stack([X, Y, Z]).astype(np.float32)

    return points_3d


def my_sad_disparity_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    max_disparity: int,
) -> np.ndarray:
    """Compute a disparity value for each pixel in img1 using the SAD metric.

    :param img1: left image
    :param img2: right image
    :param window_size: size of the window used in template matching
    :param max_disparity: maximum disparity value to search for
    :return: disparity map of same width and height as img1, in float32.
    """
    img1, img2 = np.atleast_3d(img1) / 255.0, np.atleast_3d(img2) / 255.0
    assert img1.shape == img2.shape, "Images must have the same shape."

    h, w, c = img1.shape

    if c > 1:
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
    else:
        img1_gray = img1.squeeze()
        img2_gray = img2.squeeze()

    disparity_map = np.zeros((h, w), dtype=np.float32)
    best_cost = np.full((h, w), np.inf, dtype=np.float32)

    pad = window_size // 2
    img1_padded = np.pad(img1_gray, pad, mode='constant')
    img2_padded = np.pad(img2_gray, pad, mode='constant')

    kernel = np.ones((window_size, window_size), dtype=np.float32)

    for d in range(max_disparity):
        img2_shifted = img2_padded.copy()
        if d > 0:
            img2_shifted[:, pad:-d] = img2_padded[:, pad+d:]
            img2_shifted[:, -d:] = 0

        abs_diff = np.abs(img1_padded - img2_shifted)
        sad = cv.filter2D(abs_diff, -1, kernel)
        sad_valid = sad[pad:-pad, pad:-pad]
        better_mask = sad_valid < best_cost
        disparity_map[better_mask] = d
        best_cost[better_mask] = sad_valid[better_mask]

    return disparity_map


def my_leaderboard_disparity_map(img1: np.ndarray, img2: np.ndarray):
    raise NotImplementedError(
        "Your code here – you could just call my_sad_disparity_map with carefully-chosen "
        "parameters, or get much fancier"
    )


def main(scene_folder: Path, baseline_mm: float, window_size: int, scale: float):
    # Load images
    img1 = cv.imread(str(scene_folder / "view1.png"), cv.IMREAD_COLOR)
    img2 = cv.imread(str(scene_folder / "view5.png"), cv.IMREAD_COLOR)

    if (scene_folder / "disp1.png").exists():
        true_disparities = cv.imread(str(scene_folder / "disp1.png"), cv.IMREAD_GRAYSCALE)
        with open(scene_folder / "dmin.txt") as f:
            dmin = int(f.readline())

        # Per the Middlebury middlebury-stereo data docs, disparity maps are stored relative to
        # full-resolution images and need to be scaled.
        true_disparities = true_disparities.astype(np.float32) * scale

        gt_point_cloud = plot_3d_points(
            disparity_to_3d(
                true_disparities,
                focal_length=3740 * scale,
                baseline_mm=baseline_mm,
                dmin=dmin * scale,
            ),
            colors=img1[true_disparities > 0, :].reshape(-1, 3)[:, ::-1],
        )

        o3d.visualization.draw_geometries(
            [gt_point_cloud],
            front=[0, 0, -1],
            up=[0, -1, 0],
            window_name="Ground Truth Depths",
        )

    # Heuristic: set max disparity to the smallest multiple of 16 that is larger than 1/8th the
    # image width. Note that StereoSGBM requires this to be a multiple of 16.
    max_disparity = ceil_16(img1.shape[1] / 8)

    # Compute disparity (OpenCV).
    cv_stereo_matcher = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=max_disparity,
        mode=cv.STEREO_SGBM_MODE_HH,
        blockSize=window_size,
    )
    cv_disparities = fix16_to_float32(cv_stereo_matcher.compute(img1, img2), fractional=4)

    cv_point_cloud = plot_3d_points(
        disparity_to_3d(
            cv_disparities,
            focal_length=3740 * scale,
            baseline_mm=baseline_mm,
            dmin=dmin * scale,
        ),
        colors=img1[cv_disparities > 0, :].reshape(-1, 3)[:, ::-1],
    )
    o3d.visualization.draw_geometries(
        [cv_point_cloud],
        front=[0, 0, -1],
        up=[0, -1, 0],
        window_name="OpenCV StereoSGBM best estimate",
    )

    # Compute disparity (custom implementation of SAD)
    my_sad_disparities = my_sad_disparity_map(
        img1, img2, max_disparity=max_disparity, window_size=window_size
    )

    my_sad_point_cloud = plot_3d_points(
        disparity_to_3d(
            my_sad_disparities,
            focal_length=3740 * scale,
            baseline_mm=baseline_mm,
            dmin=dmin * scale,
        ),
        colors=img1[my_sad_disparities > 0, :].reshape(-1, 3)[:, ::-1],
    )
    o3d.visualization.draw_geometries(
        [my_sad_point_cloud],
        front=[0, 0, -1],
        up=[0, -1, 0],
        window_name="My SAD depth estimate",
    )

    try:
        # Compute disparity (leaderboard)
        my_leaderboard_disparities = my_leaderboard_disparity_map(img1, img2)

        my_leaderboard_point_cloud = plot_3d_points(
            disparity_to_3d(
                my_leaderboard_disparities,
                focal_length=3740 * scale,
                baseline_mm=baseline_mm,
                dmin=dmin * scale,
            ),
            colors=img1[my_leaderboard_disparities > 0, :].reshape(-1, 3)[:, ::-1],
        )
        o3d.visualization.draw_geometries(
            [my_leaderboard_point_cloud],
            front=[0, 0, -1],
            up=[0, -1, 0],
            window_name="My leaderboard depth estimate",
        )
    except NotImplementedError:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene_folder",
        type=Path,
        help="Path to the scene folder which must at least contain view1.png, view5.png, and "
        "disp1.png.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Size of the SAD box filter.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1 / 3,
        help="Scale factor for GT disparity. "
        "Defaults to 1/3, given from the Middlebury dataset docs.",
    )
    parser.add_argument(
        "--baseline-mm",
        type=float,
        default=160,
        help="Baseline distance between cameras for view1.png and view5.png. "
        "Defaults to 160, given from the Middlebury dataset docs.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=3740,
        help="Focal length of the camera in units of pixels. "
        "Defaults to 3740, given from the Middlebury dataset docs.",
    )
    args = parser.parse_args()
    main(args.scene_folder, args.baseline_mm, args.window_size, args.scale)
