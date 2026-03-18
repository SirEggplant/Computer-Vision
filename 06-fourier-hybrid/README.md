# 06-fourier-hybrid

This assignment explores frequency domain processing using the 2D Discrete Fourier Transform. 
It includes two main demonstrations: 
generating hybrid images and interactively selecting the best‑focused image from a stack using power spectrum analysis.

**What It Does**
- Hybrid Image Generator (hybrid.py): 
Combines the low frequencies of one image with the high frequencies of another to create a hybrid image.
- Autofocus Simulator (autofocus.py): 
Analyzes a stack of images taken at different focus settings.
Lets the user click a point and adjust a radius, then uses frequency analysis to select the sharpest image in that region.
- Convolution Theorem Verification (convolution_theorem.py):
Validates that spatial‑domain convolution matches frequency‑domain multiplication using FFT. 
It runs silently and only produces output if the results do not match.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: numpy, opencv-python, matplotlib (for autofocus plotting)

**Setup**
```bash
cd 06-fourier-hybrid
pip install -r requirements.txt
```

**Running Hybrid Image Generator**
Usage: python src/hybrid.py <image1_path> <image2_path> [--low-cutoff VALUE] [--high-cutoff VALUE]
- --low-cutoff: Cutoff frequency for low‑pass filter (cycles per pixel, default: 0.05).
- --high-cutoff: Cutoff frequency for high‑pass filter (cycles per pixel, default: 0.1).

Example with provided images:
```bash
mkdir -p images
python src/hybrid.py data/elephant.png data/cheetah.png
```
- The output will be in the images folder.
- The hybrid image is saved as images/elephant_cheetah_hybrid.png.

**Running Autofocus Simulator**
Usage: python src/autofocus.py <image1> <image2> ... <imagen>[--fmin VALUE] [--fmax VALUE]
- Provide any number of image files (same scene with different focus).
- --fmin: Lower bound of frequency band (cycles per pixel, default: 0.0).
- --fmax: Upper bound (max 0.5, default: 0.5).

Example provided: 
```bash
python src/autofocus.py data/focus1.jpg data/focus2.jpg data/focus3.jpg data/focus4.jpg data/focus5.jpg data/focus6.jpg --fmin 0.02 --fmax 0.2
```

**Interactive usage:**
- A window opens showing the current best‑focused image.
- Left‑click anywhere to set the analysis region.
- Use the slider to adjust the radius.
- Press q to quit.

## Files
- src/hybrid.py                 – hybrid image generator.
- src/autofocus.py              – interactive autofocus simulator.
- src/convolution_theorem.py    – convolution theorem verification.
- src/utils.py                  – helper functions (provided with assignment).
- data/                         – sample images (elephant.png, cheetah.png, focus stack).
- data/filters                  - filters for convolution.
- images/                       - sample hybrid output.
- requirements.txt              – Python dependencies.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.