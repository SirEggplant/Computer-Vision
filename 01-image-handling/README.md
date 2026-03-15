# 01-image-handling: Image Processing Basics
This assignment implements fundamental image processing operations using NumPy and OpenCV. 
It demonstrates how to manipulate images at the pixel level, work with color channels, and apply geometric transformations.

## What It Does
The module (src/image_handling.py) provides functions for:

- Conversions between uint8 and float32 with proper scaling.
- Cropping with out‑of‑bounds handling (pixels outside the image become black).
- Scaling by factor 2 using both NumPy slicing and OpenCV.
- Horizontal mirroring (flip left‑right).
- 90° counter‑clockwise rotation.
- Color channel manipulation (swap blue/red, extract individual channels).
- Saturation scaling via HSV conversion.
- Grayscale conversion using standard luminance weights.
- Tiling the original image together with its blue, green, and red channels.

## How to Run the Demo
The included demo script loads an image, applies most of the above transformations, and displays them in separate windows.

**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python

**Run the demo**

```bash
cd 01-image-handling
pip install -r requirements.txt
cd src
python src/run_demo.py
```

This will use the default test image data/bouquet.png.
To use a different image, provide its path as a command‑line argument:

```bash
python src/run_demo.py path/to/your/image.jpg
```

## Files
- src/image_handling.py – all function implementations.
- src/run_demo.py – demonstration script.
- data/bouquet.png – sample image.
- requirements.txt – list of dependencies.