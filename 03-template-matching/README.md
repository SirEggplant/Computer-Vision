# 03-template-matching
This assignment implements object detection using template matching with multi-scale support and non-maximal suppression.

**What It Does**
- Performs template matching using OpenCV's matchTemplate with normalized cross-correlation.
- Supports multi-scale detection by scaling the image down at multiple levels.
- Converts score maps to bounding boxes at each scale.
- Applies non-maximal suppression (NMS) based on IoU to eliminate duplicate detections.
- Visualizes the detected objects by drawing rectangles on the scene.

## How to Run

**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python

**Setup**

```bash
cd 03-template-matching
pip install -r requirements.txt
```

**Running Template Matching**
Basic usage: python src/template_match.py --image <scene_image> --template <template_image> --threshold <value> --nms-threshold <value>

Example with provided images:
```bash
python src/template_match.py --image data/mario_small.jpg --template data/coin_small.png --threshold 0.7 --nms-threshold 0.3
```

Optional arguments:
- --scale-factor : Downscaling factor for each level when doing multi-scale detection (default: 0.5).
- --levels: Number of levels (image sizes) to use (default: 1, meaning only original size).

Example with multi-scale:
```bash
python src/template_match.py --image data/mario_small.jpg --template data/coin_small.png --threshold 0.7 --nms-threshold 0.3 --scale-factor 0.8 --levels 3
```

**What to Expect:**
- The script will open a window showing the scene with bounding boxes drawn around detected objects.
- The number of matches is displayed at the bottom left.
- Press any key in the window to close it.

## Files
- src/template_match.py – main object detection script.
- src/bounding_boxes.py – bounding box utilities (IoU, etc.).
- src/correlation.py – custom correlation implementation.
- src/config.py – default parameters (if provided).
- data/ – sample images (mario_small.jpg, coin_small.png).
- requirements.txt – Python dependencies.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.