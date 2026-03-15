from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

import open3d as o3d
from tqdm.auto import trange


@dataclass
class Vehicle:
    # Unique id for each vehicle to track it from frame to frame.
    vehicle_id: int = -1
    # XYZ position
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    # XYZ velocity (difference in position from previous frame)
    mvec_x: float = 0.0
    mvec_y: float = 0.0
    mvec_z: float = 0.0
    # 3D Bounding Box
    bbox_x_min: float = 0.0
    bbox_x_max: float = 1.0
    bbox_y_min: float = 0.0
    bbox_y_max: float = 1.0
    bbox_z_min: float = 0.0
    bbox_z_max: float = 1.0

    @classmethod
    def csv_header(cls):
        return ",".join(cls.__annotations__.keys())

    def csv_row(self):
        return ",".join([str(self.__dict__[field]) for field in self.__annotations__.keys()])


def write_csv_helper(file: Path, vehicles: Iterable[Vehicle]):
    # Start with header by inspecting field names of the Vehicle class; if the list of vehicles is
    # empty then we need a new default Vehicle for the header:
    with open(file, "w") as f:
        f.write(Vehicle.csv_header() + "\n")
        for v in vehicles:
            f.write(v.csv_row() + "\n")


def visualize_point_cloud(vis: o3d.visualization.Visualizer, pcd: o3d.geometry.PointCloud):
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def load_point_cloud(path_to_cloud: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(str(path_to_cloud))


def remove_ground(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd

    points = np.asarray(pcd.points)

    z_threshold = 0.1
    non_ground_mask = points[:, 2] > z_threshold

    remaining_pcd = pcd.select_by_index(np.where(non_ground_mask)[0])

    if len(remaining_pcd.points) < 50:
        return remaining_pcd

    _, inliers = remaining_pcd.segment_plane(
        distance_threshold=0.25,
        ransac_n=3,
        num_iterations=50
    )

    return remaining_pcd.select_by_index(inliers, invert=True)


def cluster_vehicles(pcd: o3d.geometry.PointCloud) -> list[o3d.geometry.PointCloud]:
    if len(pcd.points) == 0:
        return []

    labels = np.array(pcd.cluster_dbscan(
        eps=0.8,
        min_points=10,
        print_progress=False
    ))

    clusters = []
    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_indices = np.where(labels == label)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

        if len(cluster_pcd.points) >= 10:
            clusters.append(cluster_pcd)

    return clusters


def calculate_position(cluster_pcd: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(cluster_pcd.points)
    return np.mean(points, axis=0)


def is_likely_vehicle(cluster_pcd: o3d.geometry.PointCloud) -> bool:
    points = np.asarray(cluster_pcd.points)

    if len(points) < 10 or len(points) > 2000:
        return False

    bbox = cluster_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    size = max_bound - min_bound

    if size[0] > 15.0 or size[1] > 15.0 or size[2] > 6.0:
        return False
    if abs(center[0]) > 30.0 or abs(center[1]) > 30.0:
        return False
    if center[2] < 0.2 or center[2] > 5.0:
        return False

    return True


def track_vehicles(current_detections: list,
                   previous_positions: dict, frame_number: int) -> tuple[list, int]:
    MAX_TRACKING_DISTANCE = 5.0
    tracked_vehicles = []
    used_previous_ids = set()
    next_id = max(previous_positions.keys(), default=-1) + 1

    current_detections.sort(key=lambda x: len(x['cluster'].points), reverse=True)

    for detection in current_detections:
        current_pos = detection['position']
        best_match_id = -1
        min_distance = float('inf')

        for prev_id, prev_data in previous_positions.items():
            if prev_id in used_previous_ids:
                continue

            prev_pos = prev_data['position']
            distance = np.linalg.norm(current_pos - prev_pos)

            if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                min_distance = distance
                best_match_id = prev_id

        vehicle = Vehicle()
        vehicle.position_x, vehicle.position_y, vehicle.position_z = current_pos

        bbox = detection['cluster'].get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        vehicle.bbox_x_min, vehicle.bbox_y_min, vehicle.bbox_z_min = min_bound
        vehicle.bbox_x_max, vehicle.bbox_y_max, vehicle.bbox_z_max = max_bound

        if best_match_id != -1:
            vehicle.vehicle_id = best_match_id
            prev_data = previous_positions[best_match_id]
            prev_pos = prev_data['position']

            vehicle.mvec_x = current_pos[0] - prev_pos[0]
            vehicle.mvec_y = current_pos[1] - prev_pos[1]
            vehicle.mvec_z = current_pos[2] - prev_pos[2]

            used_previous_ids.add(best_match_id)
        else:
            vehicle.vehicle_id = next_id
            vehicle.mvec_x = vehicle.mvec_y = vehicle.mvec_z = 0.0
            next_id += 1

        tracked_vehicles.append(vehicle)

    return tracked_vehicles, next_id


def main(
    data_path: Path,
    output_path: Path = "perception_results",
    start_index: int = 0,
    end_index: int = -1,
    debug: bool = False,
):
    if debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600)

    if end_index < 0:
        pcd_files = list(data_path.glob("*.pcd"))
        end_index = len(pcd_files) - 1 if pcd_files else 0

    previous_vehicle_data = {}

    for frame_number in trange(start_index, end_index + 1, desc="Processing Frames"):
        vehicles = []

        pcd_path = data_path / f"{frame_number}.pcd"
        if not pcd_path.exists():
            write_csv_helper(output_path / f"{frame_number}.csv", vehicles)
            continue

        pcd = load_point_cloud(pcd_path)

        if len(pcd.points) == 0:
            write_csv_helper(output_path / f"{frame_number}.csv", vehicles)
            continue

        non_ground_pcd = remove_ground(pcd)

        if len(non_ground_pcd.points) < 10:
            write_csv_helper(output_path / f"{frame_number}.csv", vehicles)
            continue

        clusters = cluster_vehicles(non_ground_pcd)

        current_detections = []
        for cluster in clusters:
            if not is_likely_vehicle(cluster):
                continue

            position = calculate_position(cluster)
            current_detections.append({
                'cluster': cluster,
                'position': position,
                'frame': frame_number
            })

        if len(current_detections) > 6:
            current_detections.sort(key=lambda x: len(x['cluster'].points), reverse=True)
            current_detections = current_detections[:6]
        if current_detections:
            vehicles, next_vehicle_id = track_vehicles(
                current_detections, previous_vehicle_data, frame_number
            )

        previous_vehicle_data = {}
        for vehicle in vehicles:
            previous_vehicle_data[vehicle.vehicle_id] = {
                'position': np.array([vehicle.position_x, vehicle.position_y, vehicle.position_z]),
                'frame': frame_number
            }

        write_csv_helper(output_path / f"{frame_number}.csv", vehicles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        help="Directory containing .pcd files",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("perception_results"),
        help="Directory where .csv outputs will be saved",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Index of first frame",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        type=int,
        default=-1,
        help="Index of last frame (defaults to -1 for last frame)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, which turns on the visualization animation.",
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    main(**vars(args))
