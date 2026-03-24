import numpy as np
import open3d as o3d


def uint8_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert an image from uint8 format (ranging in [0,255]) to float format (ranging in [0,1]).
    """
    return image.astype(np.float32) / 255.0


def float32_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image from float format (ranging in [0,1]) to uint8 format (ranging in [0,255]).
    """
    return np.clip(np.round(image * 255), 0, 255).astype(np.uint8)


def ceil_16(value: int | float) -> int:
    """Round the given value up to the next multiple of 16"""
    return np.ceil(value / 16).astype(int) * 16


def fix16_to_float32(values: np.ndarray[np.int16], fractional=4) -> np.ndarray[np.float32]:
    return values / 2**fractional


def plot_3d_points(points_3d: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Plot a 3D point cloud using Open3D.

    Args:
        points_3d (np.ndarray): (N, 3) array of 3D coordinates.
        colors (np.ndarray): (N, 3) array of RGB colors.
        vis (o3d.visualization.O3DVisualizer, optional): optional Open3D visualizer window. Created
            by default.
    """
    # Normalize colors to [0, 1]
    colors = colors / 255.0 if colors.max() > 1 else colors

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
