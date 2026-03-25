# 11-character-recognition
This project implements an end‑to‑end Optical Character Recognition system. 
It segments handwritten text from an image and recognizes individual characters using a pre‑trained classifier. 
The result is an annotated image with bounding boxes and predicted labels.

**What It Does**
- Loads a trained model checkpoint (e.g., logistic regression or CNN).
- Segments the input image into individual character patches using classic computer vision techniques.
- Classifies each patch with the model.
- Draws bounding boxes and labels on the original image.
- Displays the annotated image in an OpenCV window.

## How to Run
**Prerequisites**
- Python 3.10+
- Required packages: torch, torchvision, opencv-python, numpy, matplotlib

**Setup**
```bash
cd 11-character-recognition
pip install -r requirements.txt
```

**Running OCR**
```bash
python src/ocr.py model/checkpoint_best.pt data/quick_brown_fox.jpg --confidence 0.5 
```
- model/checkpoint_best.pt – pre‑trained model checkpoint.
- data/quick_brown_fox.jpg – input image.
- --confidence – minimum confidence threshold for displaying a label (default 0.5).
A window will open showing the annotated image. Press any key to close.

## Files
- src/ocr.py – main OCR script.
- src/models.py – model definitions.
- src/utils.py – helper functions.
- model/checkpoint_best.pt – pre‑trained model checkpoint.
- data/ – folder containing test images.

This assignment is part of CSCI-631: Foundations of Computer Vision at Rochester Institute of Technology.