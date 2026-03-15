"""
Run a demo of all image handling functions on a given image.
Usage:
    python run_demo.py [image_path]
If image_path is omitted, defaults to 'data/bouquet.png'.
"""

import sys
import os
import cv2 as cv

# Add src directory to Python path so we can import image_handling
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import image_handling as ih


def main():
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join("..", "data", "bouquet.png")
        print(f"No image path provided, using default: {image_path}")

    # Load the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'")
        print("Check that the file exists and the path is correct.")
        return

    print(f"Loaded image: {image_path}")
    print(f"Shape: {img.shape}, dtype: {img.dtype}")

    # Define which functions to test and give them friendly names
    tests = [
        ("Original", lambda x: x),
        ("Cropped (373,1424,200,100)", lambda x: ih.crop(x, 373, 1424, 200, 100)),
        ("Scaled by half (numpy)", ih.scale_by_half_using_numpy),
        ("Horizontal mirror", ih.horizontal_mirror_image),
        ("Rotate 90° counter‑clockwise", ih.rotate_counterclockwise_90),
        ("Swap B and R", ih.swap_b_r),
        ("Grayscale", ih.grayscale),
        ("Blue channel only", ih.blues),
        ("Green channel only", ih.greens),
        ("Red channel only", ih.reds),
        ("Tile BGR", ih.tile_bgr),
        ("Scale saturation (2.0)", lambda x: ih.scale_saturation(x, 2.0)),
        ("Scale saturation (0.0)", lambda x: ih.scale_saturation(x, 0.0)),
    ]

    # Run each test and display the result
    for name, func in tests:
        result = func(img)
        cv.imshow(name, result)

    print("\nImage windows opened. Press any key while a window is active to close all.")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()