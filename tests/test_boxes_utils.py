import pytest
import torch

import numpy as np

from grimoire.core.ops.boxes import clip_boxes, validate_boxes
from grimoire.engine.boxes import BoundingBoxFormat


fixture_boxes = np.array([
    [0, 0, 10, 10],
    [-0.1, 40, 50, 50],
    [90, -90, 150, 150],
    [135, 147, 985, 1001]
])

fixture_boxes_norm = np.array([
    [0, 0, 0.5, 0.5],
    [0.5, 0.5, 1.5, 0.9],
    [-0.4, 0.6, 0.9, 0.95]
])


def test_numpy_clip_boxes():
    result = clip_boxes(fixture_boxes, image_size=(100, 100))
    expected = np.array([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ])

    np.testing.assert_array_equal(result, expected)

    result = clip_boxes(fixture_boxes_norm)
    expected = np.array([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ])

    np.testing.assert_array_equal(result, expected)


def test_torch_clip_boxes():
    result = clip_boxes(torch.from_numpy(fixture_boxes), image_size=(100, 100))
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)

    result = clip_boxes(torch.from_numpy(fixture_boxes_norm))
    expected = torch.Tensor([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)


def test_clip_boxes_wrong_imsize_shape():
    for image_size in [(100,), (100,100,100), (100,100,100,100)]:
        size = len(image_size)

        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"image_size must be of length 2, got {size}"

def test_clip_boxes_wrong_imsize_type():
    for image_size in ["image_size", 100, 100.0, False]:
        _type = type(image_size)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"Expected image_size to be of type list, tuple or torch.Tensor, got {_type}"

def test_clip_boxes_wrong_boxes_type():
    for boxes in [None, 100, 100.0, False]:
        _type = type(boxes)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == f"Expected boxes to be of type np.ndarray or torch.Tensor, got {_type}"
fixture_boxes = np.array([
    [0, 0, 10, 10],
    [-0.1, 40, 50, 50],
    [90, -90, 150, 150],
    [135, 147, 985, 1001]
])

fixture_boxes_norm = np.array([
    [0, 0, 0.5, 0.5],
    [0.5, 0.5, 1.5, 0.9],
    [-0.4, 0.6, 0.9, 0.95]
])


def test_numpy_clip_boxes():
    result = clip_boxes(fixture_boxes, image_size=(100, 100))
    expected = np.array([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ])

    np.testing.assert_array_equal(result, expected)

    result = clip_boxes(fixture_boxes_norm)
    expected = np.array([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ])

    np.testing.assert_array_equal(result, expected)


def test_torch_clip_boxes():
    result = clip_boxes(torch.from_numpy(fixture_boxes), image_size=(100, 100))
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)

    result = clip_boxes(torch.from_numpy(fixture_boxes_norm))
    expected = torch.Tensor([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)


def test_clip_boxes_wrong_imsize_shape():
    for image_size in [(100,), (100,100,100), (100,100,100,100)]:
        size = len(image_size)

        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"image_size must be of length 2, got {size}"
fixture_boxes = np.array([
    [0, 0, 10, 10],
    [-0.1, 40, 50, 50],
    [90, -90, 150, 150],
    [135, 147, 985, 1001]
])

fixture_boxes_norm = np.array([
    [0, 0, 0.5, 0.5],
    [0.5, 0.5, 1.5, 0.9],
    [-0.4, 0.6, 0.9, 0.95]
])


def test_numpy_clip_boxes():
    result = clip_boxes(fixture_boxes, image_size=(100, 100))
    expected = np.array([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ])

    np.testing.assert_array_equal(result, expected)

    result = clip_boxes(fixture_boxes_norm)
    expected = np.array([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ])

    np.testing.assert_array_equal(result, expected)


def test_torch_clip_boxes():
    result = clip_boxes(torch.from_numpy(fixture_boxes), image_size=(100, 100))
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [0, 40, 50, 50],
        [90, 0, 100, 100],
        [100, 100, 100, 100]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)

    result = clip_boxes(torch.from_numpy(fixture_boxes_norm))
    expected = torch.Tensor([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.9],
        [0, 0.6, 0.9, 0.95]
    ]).type_as(result)

    torch.testing.assert_close(result, expected)


def test_clip_boxes_wrong_imsize_shape():
    for image_size in [(100,), (100,100,100), (100,100,100,100)]:
        size = len(image_size)

        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"image_size must be of length 2, got {size}"


def test_clip_boxes_wrong_imsize_type():
    for image_size in ["image_size", 100, 100.0, False]:
        _type = type(image_size)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"Expected image_size to be of type list, tuple or torch.Tensor, got {_type}"


def test_clip_boxes_wrong_boxes_type():
    for boxes in [None, 100, 100.0, False]:
        _type = type(boxes)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == f"Expected boxes to be of type np.ndarray or torch.Tensor, got {_type}"


def test_clip_boxes_wrong_boxes_input_shape():
    for boxes in [np.array([1,2,3,4]), np.array([[1],[2],[3],[4]]), np.array([[[1,2,3,4]]])]:
        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == "Input shape must be [N, 4]"


def test_clip_boxes_wrong_imsize_type():
    for image_size in ["image_size", 100, 100.0, False]:
        _type = type(image_size)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(fixture_boxes, image_size=image_size)
        assert str(execinfo.value) == f"Expected image_size to be of type list, tuple or torch.Tensor, got {_type}"


def test_clip_boxes_wrong_boxes_type():
    for boxes in [None, 100, 100.0, False]:
        _type = type(boxes)
        with pytest.raises(TypeError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == f"Expected boxes to be of type np.ndarray or torch.Tensor, got {_type}"


def test_clip_boxes_wrong_boxes_input_shape():
    for boxes in [np.array([1,2,3,4]), np.array([[1],[2],[3],[4]]), np.array([[[1,2,3,4]]])]:
        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == "Input shape must be [N, 4]"


def test_clip_boxes_wrong_boxes_input_shape():
    for boxes in [np.array([1,2,3,4]), np.array([[1],[2],[3],[4]]), np.array([[[1,2,3,4]]])]:
        with pytest.raises(AssertionError) as execinfo:
            clip_boxes(boxes, image_size=(100, 100))
        assert str(execinfo.value) == "Input shape must be [N, 4]"


def test_validate_boxes():
    validate_boxes(np.array([[1,2,3,4]]), format=BoundingBoxFormat.XYXY)
    validate_boxes(torch.Tensor([[1,2,3,4]]), format=BoundingBoxFormat.XYXY)

    validate_boxes(np.array([[1,2,3,4]]), format=BoundingBoxFormat.XYWH)
    validate_boxes(torch.Tensor([[1,2,3,4]]), format=BoundingBoxFormat.XYWH)

    validate_boxes(np.array([[1,2,3,4]]), format=BoundingBoxFormat.CXCY)
    validate_boxes(torch.Tensor([[1,2,3,4]]), format=BoundingBoxFormat.CXCY)


def test_validate_boxes_wrong_xyxy_format():
    wrong_xyxy_boxes = np.array([
        [10, 10, 5, 15],
        [10, 10, 15, 5],
        [0, 0, 0, 0],
        [0, 0, 15, -15]
    ])

    asserts_strings = [
        "xmin must be less than xmax",
        "ymin must be less than ymax",
        "boxes cannot contain only zeros",
        "boxes cannot contain negative values"
    ]

    for wrong_xyxy, assert_string in zip(wrong_xyxy_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            validate_boxes(np.expand_dims(wrong_xyxy, axis=0), format=BoundingBoxFormat.XYXY)
        assert str(execinfo.value) == assert_string

    for wrong_xyxy, assert_string in zip(wrong_xyxy_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            wrong_xyxy_tensor = torch.from_numpy(np.expand_dims(wrong_xyxy, axis=0))
            validate_boxes(wrong_xyxy_tensor, format=BoundingBoxFormat.XYXY)
        assert str(execinfo.value) == assert_string


def test_validate_boxes_wrong_xywh_format():
    wrong_xywh_boxes = np.array([
        [10, 10, -5, 15],
        [10, 10, 15, -5],
        [0, 0, 0, 0],
        [0, 0, 15, -15]
    ])

    asserts_strings = [
        "width must be greater than 0",
        "height must be greater than 0",
        "boxes cannot contain only zeros",
        "boxes cannot contain negative values"
    ]

    for wrong_xywh, assert_string in zip(wrong_xywh_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            validate_boxes(np.expand_dims(wrong_xywh, axis=0), format=BoundingBoxFormat.XYWH)
        assert str(execinfo.value) == assert_string
    
    for wrong_xywh, assert_string in zip(wrong_xywh_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            wrong_xywh_tensor = torch.from_numpy(np.expand_dims(wrong_xywh, axis=0))
            validate_boxes(wrong_xywh_tensor, format=BoundingBoxFormat.XYWH)
        assert str(execinfo.value) == assert_string


def test_validate_boxes_wrong_cxcy_format():
    wrong_cxcy_boxes = np.array([
        [10, 10, -5, 15],
        [10, 10, 15, -5],
        [0, 0, 0, 0],
        [0, 0, 15, -15]
    ])

    asserts_strings = [
        "width must be greater than 0",
        "height must be greater than 0",
        "boxes cannot contain only zeros",
        "boxes cannot contain negative values"
    ]

    for wrong_cxcy, assert_string in zip(wrong_cxcy_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            validate_boxes(np.expand_dims(wrong_cxcy, axis=0), format=BoundingBoxFormat.CXCY)
        assert str(execinfo.value) == assert_string

    for wrong_cxcy, assert_string in zip(wrong_cxcy_boxes, asserts_strings):
        with pytest.raises(AssertionError) as execinfo:
            wrong_cxcy_tensor = torch.from_numpy(np.expand_dims(wrong_cxcy, axis=0))
            validate_boxes(wrong_cxcy_tensor, format=BoundingBoxFormat.CXCY)
        assert str(execinfo.value) == assert_string


def test_validate_boxes_unsupported_format():
    with pytest.raises(NotImplementedError) as execinfo:
        validate_boxes(np.array([[1,2,3,4]]), format="random")
    assert str(execinfo.value) == "Unsupported format random"

    with pytest.raises(NotImplementedError) as execinfo:
        validate_boxes(torch.Tensor([[1,2,3,4]]), format="XYXYCXCY")
    assert str(execinfo.value) == "Unsupported format XYXYCXCY"