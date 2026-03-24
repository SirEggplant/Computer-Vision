# 07-feature-detection
This assignment covers feature detection and matching for automated panorama stitching. 
It includes two main components: 
a custom Harris corner detector and a fully automated panorama stitcher using SIFT features and RANSAC.

**What It Does**
- Harris Corner Detector (harris_corners.py): 
Implements the Harris corner detection algorithm and compares the results side‑by‑side with OpenCV's implementation.
- Automated Panorama Stitcher (panorama.py): Uses SIFT keypoints and descriptors to find matching points between overlapping images,
then computes homographies with RANSAC and blends the images into a seamless panorama.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python (SIFT requires opencv-contrib-python; install it if cv.SIFT.create() fails)

**Setup**
```bash
cd 07-feature-detection
pip install -r requirements.txt
```

**Running Harris Corner Detector**
Usage: python src/harris_corners.py <image_path> [--blur-size N] [--block-size N] [-k K]
- <image_path>: Path to a grayscale image (e.g., data/chessboard.jpg).
- --blur-size: Size of Gaussian blur kernel applied before corner detection (default: 5).
- --block-size: Size of the neighborhood for computing the second moment matrix (default: 5).
- -k: Harris detector free parameter (default: 0.04).

Example:
```bash
python src/harris_corners.py data/paris_a.jpg --blur-size 3 --block-size 3 -k 0.05
```

**Interactive Usage:**
- A window opens showing two images side‑by‑side: OpenCV’s corners (left) and your custom implementation (right).
- A slider lets you adjust the threshold; corners above the threshold are drawn as red circles.
- Press q to close the window.

**Running Automated Panorama Stitcher**
Usage: python src/panorama.py <image1> <image2> [<image3> ...] [-b BORDER_SIZE] [-r REFERENCE_INDEX] [-o OUTPUT]
- <image1> <image2> ...: Space‑separated paths to the images to stitch (order matters).
- -b, --border-size: Blending border width in pixels (default: 50).
- -r, --reference-index: Index (0‑based) of the image used as the reference coordinate system (default: 0, i.e., the first image).
- -o, --output: Path to save the panorama (e.g., panorama.jpg). If omitted, the result is displayed in a window.

Example of stitch two images and show the result:
```bash
python src/panorama.py data/paris_a.jpg data/paris_b.jpg
```
Example of stitch three images and save to file:
```bash
python src/panorama.py data/paris_a.jpg data/paris_b.jpg data/paris_c.jpg -r 1 -o my_panorama.jpg
```

**What to Expect:**
- The script prints the number of good matches found between each image and the reference.
- If -o is not given, a window titled "Panorama" opens showing the stitched result. Press any key to close.
- If an output path is provided, the panorama is saved to that file (no window appears).
- Note: If you encounter a memory error when using the first image as reference, try a different reference index (e.g., -r 1). 
This can happen when feature matches produce an extremely large bounding box.

## Files
- src/harris_corners.py – custom Harris corner detector.
- src/panorama.py – automated panorama stitching.
- src/utils.py – helper functions.
- data/ – sample images.
- requirements.txt – Python dependencies.
- panorama_example.jpg - jpg of the result from panorama.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.