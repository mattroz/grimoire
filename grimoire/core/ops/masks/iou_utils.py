import numpy as np


def numpy_binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculate IoU metric between two binary masks.

    Args:
        mask_a (np.ndarray[H, W]): mask of shape HxW.
        mask_b (np.ndarray[H, W]): mask of shape HxW.

    Returns:
        float: intersection over union.
    """
    area_a = np.count_nonzero(mask_a == 1)
    area_b = np.count_nonzero(mask_b == 1)
    intersection = np.count_nonzero(np.logical_and(mask_a, mask_b))
    iou = intersection / (area_a + area_b - intersection + 1e-6)

    return iou


def numpy_binary_mask_iou_batch(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Calculate IoU metric between two batches of binary masks.

    Args:
        mask_a (np.ndarray[B, H, W]): mask of shape BxHxW.
        mask_b (np.ndarray[B, H, W]): mask of shape BxHxW.

    Returns:
        np.ndarray[B]: array of size B, containing IoU between corresponding masks.
    """
    area_a = np.count_nonzero(mask_a == 1, axis=(1, 2))
    area_b = np.count_nonzero(mask_b == 1, axis=(1, 2))
    intersection = np.count_nonzero(np.logical_and(mask_a, mask_b), axis=(1, 2))
    iou = intersection / (area_a + area_b - intersection + 1e-6)

    return iou




