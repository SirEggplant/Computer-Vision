import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def iou(self, other: "BBox") -> float:
        """Compute the Intersection over Union (IoU) between this bbox and another bbox."""
        x1_min, y1_min, x1_max, y1_max = self.xyxy
        x2_min, y2_min, x2_max, y2_max = other.xyxy

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = self.w * self.h
        area2 = other.w * other.h

        union_area = area1 + area2 - inter_area

        if union_area < 1e-9:
            return 0.0

        return inter_area / union_area


@dataclass
class Patch:
    """A Patch is a portion of an image. The bbox represents the location of the patch in the
    original image, and 'pixels' is the image data of the patch, matching the height and width
    specified in the bbox
    """

    bbox: BBox
    pixels: NDArray


@dataclass
class Detection:
    """A Detection is an image region that has been assigned a string label."""

    bbox: BBox
    label: str


def numpy_to_torch(images: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    if not isinstance(images, list):
        images = [images]
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    return images


def torch_to_numpy(images: torch.Tensor) -> list[np.ndarray]:
    if images.ndim == 3:
        images = torch.unsqueeze(images, dim=0)
    return list(im.permute(1, 2, 0).cpu().numpy() for im in images)


def crop_to_bbox(image: NDArray, bbox: BBox) -> NDArray:
    """Crop out a portion of an image from the given bounding box. Does not check for out of bounds
    indexing, since we assume the bbox came from something detected in the image.
    """
    x1, y1, x2, y2 = bbox.xyxy
    return image[y1:y2, x1:x2]


def annotate_bbox(
    image: NDArray, bbox: BBox, label: str = "", color: tuple[int, int, int] = (0, 255, 0)
):
    x1, y1, x2, y2 = bbox.xyxy
    cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if label:
        cv.putText(
            image, label, (x1, y1 - 5), cv.FONT_HERSHEY_PLAIN, 0.1 + (y2 - y1) / 25, color, 2
        )


def write_detections_to_file(detections: list[Detection], file_path: Path):
    assert file_path.suffix == ".json", "Can only write detections to .json files"
    det_dicts = [asdict(det) for det in detections]
    with open(file_path, "w") as f:
        json.dump(det_dicts, f)


def read_detections_from_file(file_path: Path) -> list[Detection]:
    assert file_path.suffix == ".json", "Can only read detections from .json files"
    with open(file_path, "r") as f:
        det_dicts = json.load(f)
    detections = [
        Detection(bbox=BBox(**det_dict["bbox"]), label=det_dict["label"]) for det_dict in det_dicts
    ]
    return detections
