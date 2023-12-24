import torch

import numpy as np

from torch import Tensor

from .convert import xywh_to_xyxy, cxcy_to_xyxy
from grimoire.engine.boxes import BoundingBoxFormat


def clip_boxes(boxes: np.ndarray | Tensor, image_size: list[int] | tuple[int] | None = None) -> np.ndarray | Tensor:
    """Clip boxes to image boundaries.

    Args:
        boxes (np.ndarray | Tensor[N, 4]): boxes to be clamped
        image_size (list[int] | tuple[int] | None): image size of [H, W]. Defaults to None, which means boxes are normalized.

    Returns:
        np.ndarray | Tensor[N, 4]: clamped boxes
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    if not isinstance(image_size, (list, tuple, type(None))):
        raise TypeError(f"Expected image_size to be of type list, tuple or torch.Tensor, got {type(image_size)}")
    
    if not image_size:
        image_size = (1, 1)

    assert len(image_size) == 2, f"image_size must be of length 2, got {len(image_size)}"
    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_clip_boxes(boxes, image_size)
    elif isinstance(boxes, Tensor):
        return torch_clip_boxes(boxes, image_size)
    

def validate_boxes(boxes: np.ndarray | Tensor, format: BoundingBoxFormat) -> None:
    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        numpy_validate_boxes_format(boxes, format)
    elif isinstance(boxes, Tensor):
        torch_validate_boxes_format(boxes, format)


# primitives

def torch_clip_boxes(boxes: Tensor, image_size: list[int] | tuple[int]) -> Tensor:
    """Clip boxes to image boundaries.

    Args:
        boxes (Tensor[N, 4]): boxes to be clamped
        image_size (list[int] | tuple[int]): image size of [H, W]

    Returns:
        Tensor[N, 4]: clamped boxes
    """
    h, w = image_size
    original_dtype = boxes.dtype
    boxes = boxes.clone()

    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)

    return boxes.type(original_dtype)


def numpy_clip_boxes(boxes: np.ndarray, image_size: list[int] | tuple[int]) -> np.ndarray:
    """Clip boxes to image boundaries.

    Args:
        boxes (np.ndarray[N, 4]): boxes to be clamped
        image_size (list[int] | tuple[int]): image size of [H, W]

    Returns:
        np.ndarray[N, 4]: clamped boxes
    """
    h, w = image_size
    original_dtype = boxes.dtype
    boxes = boxes.copy()

    boxes[:, 0].clip(0, w, out=boxes[:, 0])
    boxes[:, 1].clip(0, h, out=boxes[:, 1])
    boxes[:, 2].clip(0, w, out=boxes[:, 2])
    boxes[:, 3].clip(0, h, out=boxes[:, 3])

    return boxes.astype(original_dtype)


def numpy_validate_boxes_format(boxes: np.ndarray, format: BoundingBoxFormat) -> None:
    """Validate ndarray boxes elements depending on their format.

    Args:
        boxes (np.ndarray[N, 4]): boxes to be validated
        format (BoundingBoxFormat): bounding boxes format, one of [`xyxy`, `xywh`, `cxcy`]
    """

    assert not (np.count_nonzero(boxes, axis=1) == 0).any(), "boxes cannot contain only zeros"

    assert (np.sum(boxes, axis=1) > 0).all(), "boxes cannot contain negative values"

    if format == BoundingBoxFormat.XYXY:
        assert (boxes[:, 0] < boxes[:, 2]).all(), "xmin must be less than xmax"
        assert (boxes[:, 1] < boxes[:, 3]).all(), "ymin must be less than ymax"
    elif format == BoundingBoxFormat.XYWH:
        assert (boxes[:, 2] >= 0).all(), "width must be greater than 0"
        assert (boxes[:, 3] >= 0).all(), "height must be greater than 0"
    elif format == BoundingBoxFormat.CXCY:
        assert (boxes[:, 2] >= 0).all(), "width must be greater than 0"
        assert (boxes[:, 3] >= 0).all(), "height must be greater than 0"
    else:
        raise NotImplementedError(f"Unsupported format {format}")
    

def torch_validate_boxes_format(boxes: Tensor, format: BoundingBoxFormat) -> None:
    """Validate tensor boxes elements depending on their format.

    Args:
        boxes (Tensor[N, 4]): boxes to be validated
        format (BoundingBoxFormat): bounding boxes format, one of [`xyxy`, `xywh`, `cxcy`]
    """

    assert not (torch.count_nonzero(boxes, dim=1) == 0).any(), "boxes cannot contain only zeros"

    assert (torch.sum(boxes, dim=1) > 0).all(), "boxes cannot contain negative values"

    if format == BoundingBoxFormat.XYXY:
        assert (boxes[:, 0] < boxes[:, 2]).all(), "xmin must be less than xmax"
        assert (boxes[:, 1] < boxes[:, 3]).all(), "ymin must be less than ymax"
    elif format == BoundingBoxFormat.XYWH:
        assert (boxes[:, 2] >= 0).all(), "width must be greater than 0"
        assert (boxes[:, 3] >= 0).all(), "height must be greater than 0"
    elif format == BoundingBoxFormat.CXCY:
        assert (boxes[:, 2] >= 0).all(), "width must be greater than 0"
        assert (boxes[:, 3] >= 0).all(), "height must be greater than 0"
    else:
        raise NotImplementedError(f"Unsupported format {format}")