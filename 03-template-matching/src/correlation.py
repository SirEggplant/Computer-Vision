import numpy as np


def my_correlation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform correlation of the given image with the given filter or kernel. This function must
    use numpy - you may not make any calls to OpenCV or scipy. The output should exactly mimic the
    behavior or cv.filter2D with borderType=cv.BORDER_REPLICATE. That is, the output should be the
    same size as the input image, and the output should be computed "as if" the image were padded
    with a border of pixels that replicate the edge values of the input image.
    """
    # By default, use float32 for the output, to avoid overflow when summing the products. Will
    # convert back to the input image's dtype at the end.
    img = np.atleast_3d(image)
    h, w, c = img.shape
    k_h, k_w = kernel.shape

    pad_h, pad_w = k_h // 2, k_w // 2

    out = np.zeros((h, w, c), dtype=np.float32)

    img_float = img.astype(np.float32)

    for i in range(h):
        for j in range(w):
            i_start = max(0, i - pad_h)
            i_end = min(h, i + pad_h + 1)
            j_start = max(0, j - pad_w)
            j_end = min(w, j + pad_w + 1)

            neighborhood = img_float[i_start:i_end, j_start:j_end, :]

            k_i_start = pad_h - (i - i_start)
            k_i_end = k_i_start + (i_end - i_start)
            k_j_start = pad_w - (j - j_start)
            k_j_end = k_j_start + (j_end - j_start)

            kernel_region = kernel[k_i_start:k_i_end, k_j_start:k_j_end]

            for ch in range(c):
                out[i, j, ch] = np.sum(neighborhood[:, :, ch] * kernel_region)

    if image.ndim == 2:
        out = out.squeeze()

    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)

    # Convert back to the input image's dtype, rounding to the nearest integer and clipping the
    # range if the input image's dtype is some kind of integer type
    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    return out.astype(image.dtype)
