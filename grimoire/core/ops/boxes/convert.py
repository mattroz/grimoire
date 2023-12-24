import torch

import numpy as np

from torch import Tensor

from grimoire.engine.checks import assert_shape


def xyxy_to_xywh(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height] format.

    Args:
        boxes (np.ndarray | Tensor): [xmin, ymin, xmax, ymax] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [xmin, ymin, width, height] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_xyxy_to_xywh(boxes)
    elif isinstance(boxes, Tensor):
        return torch_xyxy_to_xywh(boxes)


def xyxy_to_cxcy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [xmin, ymin, xmax, ymax] to [cx, cy, width, height] format.

    Args:
        boxes (np.ndarray | Tensor): [xmin, ymin, xmax, ymax] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [cx, cy, width, height] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_xyxy_to_cxcy(boxes)
    elif isinstance(boxes, Tensor):
        return torch_xyxy_to_cxcy(boxes)


def xywh_to_xyxy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax] format.

    Args:
        boxes (np.ndarray | Tensor): [xmin, ymin, width, height] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [xmin, ymin, xmax, ymax] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"
    
    if isinstance(boxes, np.ndarray):
        return numpy_xywh_to_xyxy(boxes)
    elif isinstance(boxes, Tensor):
        return torch_xywh_to_xyxy(boxes)


def xywh_to_cxcy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [xmin, ymin, width, height] to [cx, cy, width, height] format.

    Args:
        boxes (np.ndarray | Tensor): [xmin, ymin, width, height] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [cx, cy, width, height] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_xywh_to_cxcy(boxes)
    elif isinstance(boxes, Tensor):
        return torch_xywh_to_cxcy(boxes)
    

def cxcy_to_xyxy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [cx, cy, width, height] to [xmin, ymin, xmax, ymax] format.

    Args:
        boxes (np.ndarray | Tensor): [cx, cy, width, height] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [xmin, ymin, xmax, ymax] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_cxcy_to_xyxy(boxes)
    elif isinstance(boxes, Tensor):
        return torch_cxcy_to_xyxy(boxes)
    

def cxcy_to_xywh(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Converts boxes from [cx, cy, width, height] to [xmin, ymin, width, height] format.

    Args:
        boxes (np.ndarray | Tensor): [cx, cy, width, height] boxes of size [N, 4].

    Raises:
        TypeError: If boxes is not of type np.ndarray or torch.Tensor.

    Returns:
        np.ndarray | Tensor: [xmin, ymin, width, height] boxes of size [N, 4].
    """

    if not isinstance(boxes, (np.ndarray, Tensor)):
        raise TypeError(f"Expected boxes to be of type np.ndarray or torch.Tensor, got {type(boxes)}")

    assert len(boxes.shape) == 2 and boxes.shape[1] == 4, "Input shape must be [N, 4]"

    if isinstance(boxes, np.ndarray):
        return numpy_cxcy_to_xywh(boxes)
    elif isinstance(boxes, Tensor):
        return torch_cxcy_to_xywh(boxes)


# primitives


def numpy_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height] format.

    Args:
        boxes (np.ndarray[N, 4]): [xmin, ymin, xmax, ymax] format.

    Returns:
        boxes (np.ndarray[N, 4]): [xmin, ymin, width, height] format.
    """
    boxes = boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    
    return boxes


def numpy_xyxy_to_cxcy(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [xmin, ymin, xmax, ymax] to [cx, cy, width, height] format.

    Args:
        boxes (np.ndarray[N, 4]): [xmin, ymin, xmax, ymax] format.

    Returns:
        boxes (np.ndarray[N, 4]): [cx, cy, width, height] format.
    """
    boxes = boxes.copy()
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] = boxes[:, 0] + width / 2
    boxes[:, 1] = boxes[:, 1] + height / 2
    boxes[:, 2] = width
    boxes[:, 3] = height

    return boxes


def numpy_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax] format.

    Args:
        boxes (np.ndarray[N, 4]): [xmin, ymin, width, height] format.

    Returns:
        boxes (np.ndarray[N, 4]): [xmin, ymin, xmax, ymax] format.
    """
    boxes = boxes.copy()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    return boxes


def numpy_xywh_to_cxcy(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [xmin, ymin, width, height] to [cx, cy, width, height] format.

    Args:
        boxes (np.ndarray[N, 4]): [xmin, ymin, width, height] format.

    Returns:
        boxes (np.ndarray[N, 4]): [cx, cy, width, height] format.
    """
    boxes = boxes.copy()

    boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

    return boxes


def numpy_cxcy_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [cx, cy, width, height] to [xmin, ymin, xmax, ymax] format.

    Args:
        boxes (np.ndarray[N, 4]): [cx, cy, width, height] format.

    Returns:
        boxes (np.ndarray[N, 4]): [xmin, ymin, xmax, ymax] format.
    """
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    return boxes


def numpy_cxcy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Converts boxes from [cx, cy, width, height] to [xmin, ymin, width, height] format.

    Args:
        boxes (np.ndarray[N, 4]): [cx, cy, width, height] format.

    Returns:
        boxes (np.ndarray[N, 4]): [xmin, ymin, width, height] format.
    """
    boxes = boxes.copy()
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

    return boxes


# TORCH BOXES PRIMITIVES

def torch_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [xmin, ymin, xmax, ymax] format to [xmin, ymin, width, height] format.
   
    Args:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, xmax, ymax] format.

    Returns:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, width, height] format.
    """
    original_dtype = boxes.dtype
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1  
    h = y2 - y1
    boxes = torch.stack((x1, y1, w, h), dim=-1)

    return boxes.type(original_dtype)


def torch_xyxy_to_cxcy(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [xmin, ymin, xmax, ymax] format to [cx, cy, width, height] format.
    
    Args:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, xmax, ymax] format format.

    Returns:
        boxes (Tensor(N, 4)): boxes in [cx, cy, width, height] format.
    """
    original_dtype = boxes.dtype
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes.type(original_dtype)


def torch_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [xmin, ymin, width, height] format to [xmin, ymin, xmax, ymax] format.
    (x, y) refers to top left of bounding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, width, height] format.

    Returns:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, xmax, ymax] format.
    """
    original_dtype = boxes.dtype
    x, y, w, h = boxes.unbind(-1)
    boxes = torch.stack([x, y, x + w, y + h], dim=-1)

    return boxes.type(original_dtype)


def torch_xywh_to_cxcy(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [xmin, ymin, width, height] format to [cx, cy, width, height] format.
    Args:
        boxes (Tensor[N, 4]): boxes in [xmin, ymin, width, height] format.

    Returns:
        boxes (Tensor[N, 4]): boxes in [cx, cy, width, height] format.
    """
    original_dtype = boxes.dtype
    x, y, w, h = boxes.unbind(-1)
    cx = x + w / 2
    cy = y + h / 2
    boxes = torch.stack([cx, cy, w, h], dim=-1)

    return boxes.type(original_dtype)


def torch_cxcy_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [cx, cy, width, height] format to [xmin, ymin, xmax, ymax] format.
    Args:
        boxes (Tensor[N, 4]): boxes in [cx, cy, width, height] format format.

    Returns:
        boxes (Tensor(N, 4)): boxes in [xmin, ymin, xmax, ymax] format.
    """
    original_dtype = boxes.dtype
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes.type(original_dtype)


def torch_cxcy_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts boxes from [cx, cy, width, height] format to [xmin, ymin, width, height] format.
    Args:
        boxes (Tensor[N, 4]): boxes in [cx, cy, width, height] format format.

    Returns:
        boxes (Tensor(N, 4)): boxes in [xmin, ymin, width, height] format.
    """
    original_dtype = boxes.dtype
    cx, cy, w, h = boxes.unbind(-1)
    x = cx - 0.5 * w
    y = cy - 0.5 * h

    boxes = torch.stack((x, y, w, h), dim=-1)

    return boxes.type(original_dtype)



