from enum import Enum


class BoundingBoxFormat(Enum):
    """Coordinate format of a bounding box.

    Available formats: [`xyxy`, `xywh`, `cxcy`]
    """

    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCY = "CXCY"