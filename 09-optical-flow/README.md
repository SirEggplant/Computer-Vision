# 09-optical-flow
This assignment implements a coarse‑to‑fine optical flow algorithm using the Lucas‑Kanade method on a Gaussian pyramid. It computes motion between two consecutive frames and visualizes the results with quiver plots and HSV flow maps.

**What It Does**
- Loads two images from a consecutive frames from a video sequence.
- Builds a Gaussian pyramid for each image.
- At each pyramid level, warps the first image toward the second using the flow estimated from the coarser level.
- Solves the optical flow constraint equation with a local window (Lucas‑Kanade) to compute residual motion.
- Refines the flow estimate by propagating it up the pyramid and scaling appropriately.
- Performs a forward‑reverse consistency check to mark occluded regions.
- Visualizes the final flow field with:
  - A quiver plot (arrows on a regular grid).
  - An HSV color map where hue encodes direction and saturation encodes speed.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python, matplotlib

**Setup**
```bash
cd 09-optical-flow
pip install -r requirements.txt
```
**Data**
- Download the Middlebury optical flow dataset from https://vision.middlebury.edu/flow/data/
- Extract the folder and place it inside the data/ directory.
- Each scene folder contains two frames to compute flow between.

**Running the Script**
```bash
python src/coarse_to_fine_optical_flow.py <image1> <image2> [--levels N] [--window_size N] [--alpha A] [--goodness-threshold T]
```
- <image1>: path to the first frame (e.g., data/Urban2/frame10.png).
- <image2>: path to the second frame (e.g., data/Urban2/frame11.png).
- --levels: number of pyramid levels (default: 5).
- --window_size: size of the Lucas‑Kanade window (odd integer, default: 7).
- --alpha: regularization parameter for ridge regression (default: 1e-3).
- --goodness-threshold: threshold for forward‑reverse consistency check (default: 2.0).

Example:

```bash
python src/coarse_to_fine_optical_flow.py data/Beanbags/frame10.png data/Beanbags/frame11.png --levels 5 --window_size 5
```

**What to Expect**
Two matplotlib windows will open:
- Quiver plot – arrows showing motion vectors (direction and magnitude) at regular grid points (spacing 10 pixels).
- HSV flow map – color‑coded dense visualization:
- Hue = direction of motion (e.g., red = right, cyan = left, green = up, purple = down).
- Saturation = magnitude (bright = fast).
- Press any key or close the windows to exit.
- If the algorithm works correctly, moving objects will appear as coherent regions with consistent colors and arrows.

## Files
- src/coarse_to_fine_optical_flow.py – main optical flow script.
- src/utils.py – helper functions (pyramid construction, visualization, etc.).
- data/ – folder containing the Middlebury dataset scenes.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.