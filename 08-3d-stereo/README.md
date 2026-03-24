# 08-3d-stereo
This assignment implements stereo vision to compute disparity maps from rectified image pairs and reconstruct 3D point clouds. 
It includes a custom Sum of Absolute Differences (SAD) stereo matching algorithm and uses OpenCV's StereoSGBM as a reference. 
The results are visualized as interactive 3D point clouds using Open3D.

**What It Does**
- Loads left and right stereo images from a Middlebury dataset scene folder.
- Computes disparity maps using:
  - A custom Sum of Absolute Differences implementation with a sliding window.
  - OpenCV's StereoSGBM (Semi-Global Block Matching) as a reference.
- Converts disparity maps to 3D point clouds using camera parameters.
- Visualizes result as interactive 3D point clouds.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python, open3d

**Setup**
```bash
cd 08-3d-stereo
pip install -r requirements.txt
```

**Data**
- There is a given example but other images are also available to download.
- Download other scenes from the Middlebury stereo dataset at https://vision.middlebury.edu/stereo/data/scenes2005/
- Choose a scene (e.g., "Art") and download the "FullSize 2 views" version.
- Extract the folder and place it inside data folder.
- The folder should contain: 
  - view1.png as the left image
  - view5.png as the right image
  - dmin.txt as the dmin value for the scene
  - disp1.png as the left ground truth disparity (optional)
  - disp5.png as the right ground truth disparity (optional)

**Running the Script**
```bash
python src/stereo.py <scene_folder> [--window-size N] [--scale SCALE] [--baseline-mm B] [--focal-length F]
```
- <scene_folder>: Path to the scene folder (e.g., data/Art).
- --window-size: Size of the SAD window (default: 7).
- --scale: Scale factor for ground truth disparity (default: 1/3).
- --baseline-mm: Baseline between cameras in millimeters (default: 160).
- --focal-length: Focal length in pixels (default: 3740).

Example:
```bash
python src/stereo.py data/Art
```

**Usage:**
The script will display three (or more) Open3D windows sequentially:
- Ground truth point cloud (if disp1.png exists)
- OpenCV StereoSGBM result (disparity computed from left and right images)
- Your custom SAD result (disparity computed from left and right images)
- Each window shows the 3D point cloud, colored by the left image pixels.
- Close each window by pressing q or closing the window to proceed to the next.

## Files
- src/stereo.py – main stereo vision script.
- src/utils.py – helper functions (provided with assignment).
- data/ – folder containing Middlebury scene folders.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.