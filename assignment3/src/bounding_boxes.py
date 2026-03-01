from typing import Union

Number = Union[int, float]
BBoxType = tuple[Number, Number, Number, Number]


def bbox_xyxy_to_xywh(bbox: BBoxType) -> BBoxType:
    """Converts a bounding box from (x1, y1, x2, y2) format to (x, y, w, h) format."""
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)


def bbox_xywh_to_xyxy(bbox: BBoxType) -> BBoxType:
    """Converts a bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format."""
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return (x1, y1, x2, y2)


def bbox_xywh_iou(bbox_a: BBoxType, bbox_b: BBoxType) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes. The bounding boxes are
    given in (x, y, w, h) format.
    """
    a_x1, a_y1, a_x2, a_y2 = bbox_xywh_to_xyxy(bbox_a)
    b_x1, b_y1, b_x2, b_y2 = bbox_xywh_to_xyxy(bbox_b)

    x1 = max(a_x1, b_x1)
    y1 = max(a_y1, b_y1)
    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area
