import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    # set image as float then dividing it by 255.0 to make the range [0, 1]
    return image.astype(np.float32) / 255.0


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    # make sure image is in the range of 0 and 1, then multiple by 255 and set the type to uint8
    clipped_image = np.clip(image, 0.0, 1.0)
    return np.round(clipped_image * 255).astype(np.uint8)


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    # create blank imagine with the cropped height and width
    cropped = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
    
    src_y_start = max(y, 0)
    src_y_end = min(y + h, image.shape[0])
    src_x_start = max(x, 0)
    src_x_end = min(x + w, image.shape[1])

    dst_y_start = src_y_start - y
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    cropped[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
        src_y_start:src_y_end, src_x_start:src_x_end]
    return cropped


def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    return image[::2, ::2, :]


def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    new_size = (image.shape[1] // 2, image.shape[0] // 2)
    return cv.resize(image, new_size, interpolation=cv.INTER_NEAREST)


def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    return image[:, ::-1, :]


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    return np.transpose(image, (1, 0, 2))[::-1, :, :]


def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    return image[:, :, [2, 1, 0]]


def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    result = np.zeros_like(image)
    result[:, :, 0] = image[:, :, 0]
    return result


def greens(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the green
    channel
    """
    result = np.zeros_like(image)
    result[:, :, 1] = image[:, :, 1]
    return result


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    result = np.zeros_like(image)
    result[:, :, 2] = image[:, :, 2]
    return result


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * scale, 0, 255)
    return cv.cvtColor(hsv_image.astype(np.uint8), cv.COLOR_HSV2BGR)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    gray = (0.114 * image[:, :, 0] + 0.587 * image[
        :, :, 1] + 0.299 * image[:, :, 2]).astype(np.uint8)
    return np.stack((gray, gray, gray), axis=-1)


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    H, W, C = image.shape
    canvas = np.zeros((2*H, 2*W, C), dtype=image.dtype)
    canvas[0:H, 0:W] = image
    canvas[0:H, W:2*W] = blues(image)
    canvas[H:2*H, 0:W] = greens(image)
    canvas[H:2*H, W:2*W] = reds(image)
    return canvas
