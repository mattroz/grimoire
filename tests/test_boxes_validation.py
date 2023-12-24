import pytest
import torch

import numpy as np

from grimoire.core.ops.boxes import clip_boxes, validate_boxes


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