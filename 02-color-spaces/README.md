# 02-color-spaces
This assignment explores two core computer vision topics: 
color space manipulation for image enhancement and projective geometry for panorama creation. 
The first part implements an underwater image enhancement pipeline using gain adjustment and partial histogram equalization. 
The second part involves manually selecting corresponding points to compute a homography and blend two images into a seamless panorama.

**What It Does**

Part 1: Underwater Image Enhancement (underwater.py)
- Applies per-channel gain to compensate for the blue/green tint caused by water absorption (red channel boosted, blue/green reduced).
- Converts the image to LAB color space and performs partial histogram equalization on the lightness (L) channel to improve contrast.
- The degree of equalization is controlled by an alpha parameter (0 = no change, 1 = full equalization).
- Saves the enhanced image to disk.

Part 2: Panorama Stitching (panorama.py)
- Displays two images side-by-side and lets you click at least 4 corresponding points in each image.
- Computes a homography that maps points from the second image to the first.
- Warps both images into a common coordinate system large enough to hold the full panorama.
- Blends the overlapping regions using weighted averaging (alpha blending) to avoid hard edges.
- Shows the final stitched panorama.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python

**Setup**
```bash
cd 02-color-spaces
pip install -r requirements.txt
```

**Running Underwater Enhancement**

Basic usage: python src/underwater.py --input <image_path> --output <output_path>

Example with provided images:
```bash
python src/underwater.py --input data/underwater01.jpg --output enhanced01.jpg
```

Optional parameters:
- --gain-b , --gain-g , --gain-r : multipliers for blue, green, and red channels (defaults: 0.9, 0.9, 1.3).
- --alpha : blending factor for partial histogram equalization (default: 0.7).

Example with custom parameters:
```bash
python src/underwater.py --input data/underwater02.jpg --output my_enhanced.jpg --gain-r 1.5 --alpha 0.5
```

The enhanced image will be saved to the location specified by --output.

**Running Panorama Stitching**

The script src/panorama.py loads two images and lets you manually select matching points.

Usage: python src/panorama.py --image1 <path> --image2 <path> --output <output_path> [--border-size <pixels>]

Example with provided images:
```bash
python src/panorama.py --image1 data/paris_a.jpg --image2 data/paris_b.jpg --output panorama.jpg
```

The optional --border-size argument controls the size of the blending border (default: 50).

**Interactive Workflow:**
1. Two windows will appear: "left image" and "right image".
2. Click on a distinctive point in the left image (e.g., a corner of a building).
3. Immediately click on the corresponding point in the right image.
4. Repeat for at least 4 points. The more points you select (and the more accurately), the better the homography.
5. When finished, press 'Enter' in one of the image windows.
6. The script will compute the homography, warp the images, blend them, and save the final panorama to the specified output file.
7. The result will be saved as a jpg.

Tip: Choose points that are spread out across the scene and not collinear for best results.

## Files
- src/underwater.py – underwater enhancement implementation.
- src/panorama.py – panorama stitching with manual point selection.
- data/ – sample images
- requirements.txt – list of Python dependencies.
- panorama_example.jpg - jpg of the result from panorama.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.