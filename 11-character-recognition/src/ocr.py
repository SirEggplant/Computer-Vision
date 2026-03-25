import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from models import get_model
from utils import Detection, Patch, BBox


class CharacterSegmenter:
    """This class is responsible for taking an image and segmenting it into small square patches,
    one per character, to be fed into the character recognition system.
    """
    def __init__(self, patch_size: tuple[int, int] = (28, 28)):
        self.patch_size = patch_size

    def __call__(self, image: NDArray[np.uint8]) -> list[Patch]:
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        MAX_SIZE = 800
        scale = 1.0
        if h > MAX_SIZE or w > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)

        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        h, w = binary.shape
        center = binary[h//4:3*h//4, w//4:3*w//4]
        if np.mean(center) > 127:
            binary = cv.bitwise_not(binary)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)

            if h < 10 or w < 3:
                continue
            if h > 150 or w > 100:
                continue

            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1] // 15, b[0]))

        patches = []

        for x, y, w, h in boxes:

            char_img = binary[y:y+h, x:x+w]

            size = max(h, w) + 4
            square = np.zeros((size, size), dtype=np.uint8)
            y_off = (size - h) // 2
            x_off = (size - w) // 2
            square[y_off:y_off+h, x_off:x_off+w] = char_img

            resized = cv.resize(square, self.patch_size, interpolation=cv.INTER_AREA)

            normalized = resized.astype(np.float32) / 255.0

            if not np.isclose(scale, 1.0, rtol=1e-09, atol=1e-09):
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
            bbox = BBox(x, y, w, h)

            patches.append(Patch(bbox=bbox, pixels=normalized))

        return patches


class OpticalCharacterRecognition:
    """Character-based OCR class, implementing a two-stage segment-then-classify pipeline."""

    def __init__(
        self,
        segmenter: CharacterSegmenter,
        classifier: nn.Module,
        class_labels: list[str],
        confidence_threshold: float = 0.9,
        device: str | torch.device = "cpu",
    ):
        self.segmenter = segmenter
        self.classifier = classifier.eval().to(device)
        self.class_labels = class_labels
        self.confidence_threshold = confidence_threshold
        self.device = device

    @staticmethod
    def new_from_checkpoint(checkpoint: Path, **kwargs) -> "OpticalCharacterRecognition":
        """Instantiate a new OCR model from a checkpoint file."""
        ckpt = torch.load(checkpoint, map_location="cpu")
        classes = ckpt["classes"]
        model = get_model(ckpt["model_slug"], num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        return OpticalCharacterRecognition(
            segmenter=CharacterSegmenter(),
            classifier=model,
            class_labels=classes,
            **kwargs,
        )

    def process_image(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Take an image containing handwritten text along with a bounding box marking the rough
        boundaries of the 'document' in the image, then return a list of detected characters.
        """
        patches = self.segmenter(image)
        torch_data = (
            torch.from_numpy(np.array([np.atleast_3d(patch.pixels) for patch in patches]))
            .to(self.device)
            .permute(0, 3, 1, 2)
        )
        predictions = torch.softmax(self.classifier(torch_data), dim=-1)
        ids = torch.argmax(predictions, dim=-1)
        confidence = torch.max(predictions, dim=-1).values
        detections = []
        for patch, idx, score in zip(patches, ids, confidence):
            if score > self.confidence_threshold:
                detections.append(Detection(bbox=patch.bbox, label=self.class_labels[idx]))

        return detections

    def annotate(self, image: NDArray[np.uint8]) -> np.ndarray:
        detections = self.process_image(image)
        annotated = image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox.xyxy
            cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                annotated,
                detection.label,
                (x1, y1 - 5),
                cv.FONT_HERSHEY_PLAIN,
                0.1 + (y2 - y1) / 25,
                (0, 255, 0),
                2,
            )
        return annotated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR model")
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint containing model weights plus all metadata to "
        "instantiate the model and label things.",
    )
    parser.add_argument("image", type=Path, help="Path to the image to annotate.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold in the class label to call it a valid detection.",
    )
    args = parser.parse_args()

    im = cv.imread(str(args.image))
    if im is None:
        raise FileNotFoundError(args.image)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    model = get_model(ckpt["model_slug"], num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])

    ocr = OpticalCharacterRecognition(
        segmenter=CharacterSegmenter(),
        classifier=model,
        class_labels=classes,
        confidence_threshold=args.confidence,
    )
    annotated = ocr.annotate(im)

    cv.imshow("Annotated", annotated)
    cv.waitKey(0)
    cv.destroyAllWindows()
