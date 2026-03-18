# 05-shape-detection
This script detects shapes of a specified size, color, and type in an image. 
It uses Otsu thresholding, color‑based segmentation in HSV space, connected components analysis, and moment‑based shape classification.

**What It Does**
- Loads an image and isolates regions of a given color by thresholding on hue.
- Finds connected components (individual shapes) and computes their moments.
- Classifies each shape as circle, wedge, rectangle, or cross using roundedness and rotational symmetry tests.
- Separates shapes into "small" and "large" categories using Otsu’s method on the area histogram.
- Returns the centroid coordinates of all shapes that match the requested size, color, and type.
- Annotates the original image with a crosshair at each detected location and displays the result.

## How to Run

**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python

**Setup**
```bash
cd 05-shape-detection
pip install -r requirements.txt
```

**Running the Script**
Usage: python src/find_shapes.py <image_path> <size> <color> <shape> [--min-area <pixels>] [--hue-tolerance <degrees>]

Positional Arguments:
- image_path: Path to the input image (e.g., data/shapes.png).
- size: Either large or small.
- color: red, yellow, green, cyan, blue, or magenta.
- shape: circle, wedge, rectangle, or cross.

Optional Arguments:
  --min-area    Minimum area (in pixels) for a component to be considered a shape (default: 10).
  --hue-tolerance  Allowed deviation from the target hue when thresholding (default: 10).

Examples:

Find all small red circles in the sample image:
```bash
python src/shape_finder.py data/shapes.png small red circle
```
Find large blue rectangles:
```bash
python src/shape_finder.py data/shapes.png large blue rectangle
```
Find large green wedges:
```bash
python src/shape_finder.py data/shapes.png large green wedge
```

**What to Expect**
- The script will open a window showing the original image with a crosshair drawn at the centroid of each detected shape.
- Press any key to close the window.

## Files
- src/find_shapes.py – main shape detection script.
- src/bounding_boxes.py – (may be used for IoU, but not directly here).
- data/shapes.png – sample test image.
- requirements.txt – Python dependencies.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.