import sys
sys.path.append("../")

from grimoire.core.ops.boxes.convert import *

# fixtures

fixture_boxes_xyxy = np.array([
    [0, 0, 10, 10],
    [40, 40, 50, 50],
    [90, 90, 100, 100],
])

fixture_boxes_xyxy_norm = np.array([
    [0, 0, 0.5, 0.5],
    [0.5, 0.5, 0.9, 0.9],
    [0.4, 0.6, 0.9, 0.95]
])

fixture_boxes_xywh = np.array([
    [0, 0, 10, 10],
    [40, 40, 50, 50],
    [90, 90, 100, 100],
])

fixture_boxes_xywh_norm = np.array([
    [0, 0, 0.3, 0.4],
    [0.5, 0.5, 0.25, 0.45],
    [0.3, 0.7, 0.65, 0.2],
])

fixture_boxes_cxcy = np.array([
    [5, 5, 10, 10],
    [40, 40, 50, 50],
    [90, 90, 100, 100],
])

fixture_boxes_cxcy_norm = np.array([
    [0.3, 0.3, 0.3, 0.3],
    [0.3, 0.5, 0.6, 0.6],
    [0.5, 0.5, 0.7, 0.7]
])

# tests

def test_numpy_xyxy_to_xywh():
    result = numpy_xyxy_to_xywh(fixture_boxes_xyxy)
    expected = np.array([
        [0, 0, 10, 10],
        [40, 40, 10, 10],
        [90, 90, 10, 10],
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_xyxy_to_xywh(fixture_boxes_xyxy_norm)
    expected = np.array([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 0.4, 0.4],
        [0.4, 0.6, 0.5, 0.35]
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_numpy_xyxy_to_cxcy():
    result = numpy_xyxy_to_cxcy(fixture_boxes_xyxy)
    expected = np.array([
        [5, 5, 10, 10],
        [45, 45, 10, 10],
        [95, 95, 10, 10],
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_xyxy_to_cxcy(fixture_boxes_xyxy_norm)
    expected = np.array([
        [0.25, 0.25, 0.5, 0.5],
        [0.7, 0.7, 0.4, 0.4],
        [0.65, 0.7750, 0.5, 0.35]
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_numpy_xywh_to_xyxy():
    result = numpy_xywh_to_xyxy(fixture_boxes_xywh)
    expected = np.array([
        [0, 0, 10, 10],
        [40, 40, 90, 90],
        [90, 90, 190, 190]
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_xywh_to_xyxy(fixture_boxes_xywh_norm)
    expected = np.array([
        [0, 0, 0.3, 0.4],
        [0.5, 0.5, 0.75, 0.95],
        [0.3, 0.7, 0.95, 0.9]
    ])

    np.testing.assert_array_almost_equal(result, expected)    


def test_numpy_xywh_to_cxcy():
    result = numpy_xywh_to_cxcy(fixture_boxes_xywh)
    expected = np.array([
        [5, 5, 10, 10],
        [65, 65, 50, 50],
        [140, 140, 100, 100]
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_xywh_to_cxcy(fixture_boxes_xywh_norm)
    expected = np.array([
        [0.15, 0.2, 0.3, 0.4],
        [0.625, 0.725, 0.25, 0.45],
        [0.625, 0.8, 0.65, 0.2]
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_numpy_cxcy_to_xyxy():
    result = numpy_cxcy_to_xyxy(fixture_boxes_cxcy)
    expected = np.array([
        [0, 0, 10, 10],
        [15, 15, 65, 65],
        [40, 40, 140, 140]
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_cxcy_to_xyxy(fixture_boxes_cxcy_norm)
    expected = np.array([
        [0.15, 0.15, 0.45, 0.45],
        [0, 0.2, 0.6, 0.8],
        [0.15, 0.15, 0.85, 0.85],
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_numpy_cxcy_to_xywh():
    result = numpy_cxcy_to_xywh(fixture_boxes_cxcy)
    expected = np.array([
        [0, 0, 10, 10],
        [15, 15, 50, 50],
        [40, 40, 100, 100]
    ])

    np.testing.assert_array_equal(result, expected)

    result = numpy_cxcy_to_xywh(fixture_boxes_cxcy_norm)
    expected = np.array([
        [0.15, 0.15, 0.3, 0.3],
        [0, 0.2, 0.6, 0.6],
        [0.15, 0.15, 0.7, 0.7],
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_torch_xyxy_to_xywh():
    fixture_boxes_xyxy_torch = torch.from_numpy(fixture_boxes_xyxy)
    result = torch_xyxy_to_xywh(fixture_boxes_xyxy_torch)
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [40, 40, 10, 10],
        [90, 90, 10, 10],
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_xyxy_torch_norm = torch.from_numpy(fixture_boxes_xyxy_norm)
    result = torch_xyxy_to_xywh(fixture_boxes_xyxy_torch_norm)
    expected = torch.Tensor([
        [0, 0, 0.5, 0.5],
        [0.5, 0.5, 0.4, 0.4],
        [0.4, 0.6, 0.5, 0.35]
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_torch_xyxy_to_cxcy():
    fixture_boxes_xyxy_torch = torch.from_numpy(fixture_boxes_xyxy)
    result = torch_xyxy_to_cxcy(fixture_boxes_xyxy_torch)
    expected = torch.Tensor([
        [5, 5, 10, 10],
        [45, 45, 10, 10],
        [95, 95, 10, 10],
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_xyxy_torch_norm = torch.from_numpy(fixture_boxes_xyxy_norm)
    result = torch_xyxy_to_cxcy(fixture_boxes_xyxy_torch_norm)
    expected = torch.Tensor([
        [0.25, 0.25, 0.5, 0.5],
        [0.7, 0.7, 0.4, 0.4],
        [0.65, 0.7750, 0.5, 0.35]
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_torch_xywh_to_xyxy():
    fixture_boxes_xywh_torch = torch.from_numpy(fixture_boxes_xywh)
    result = torch_xywh_to_xyxy(fixture_boxes_xywh_torch)
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [40, 40, 90, 90],
        [90, 90, 190, 190]
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_xywh_torch_norm = torch.from_numpy(fixture_boxes_xywh_norm)
    result = torch_xywh_to_xyxy(fixture_boxes_xywh_torch_norm)
    expected = torch.Tensor([
        [0, 0, 0.3, 0.4],
        [0.5, 0.5, 0.75, 0.95],
        [0.3, 0.7, 0.95, 0.9]
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_torch_xywh_to_cxcy():
    fixture_boxes_xywh_torch = torch.from_numpy(fixture_boxes_xywh)
    result = torch_xywh_to_cxcy(fixture_boxes_xywh_torch)
    expected = torch.Tensor([
        [5, 5, 10, 10],
        [65, 65, 50, 50],
        [140, 140, 100, 100]
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_xywh_torch_norm = torch.from_numpy(fixture_boxes_xywh_norm)
    result = torch_xywh_to_cxcy(fixture_boxes_xywh_torch_norm)
    expected = torch.Tensor([
        [0.15, 0.2, 0.3, 0.4],
        [0.625, 0.725, 0.25, 0.45],
        [0.625, 0.8, 0.65, 0.2]
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_torch_cxcy_to_xyxy():
    fixture_boxes_cxcy_torch = torch.from_numpy(fixture_boxes_cxcy)
    result = torch_cxcy_to_xyxy(fixture_boxes_cxcy_torch)
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [15, 15, 65, 65],
        [40, 40, 140, 140]
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_cxcy_torch_norm = torch.from_numpy(fixture_boxes_cxcy_norm)
    result = torch_cxcy_to_xyxy(fixture_boxes_cxcy_torch_norm)
    expected = torch.Tensor([
        [0.15, 0.15, 0.45, 0.45],
        [0, 0.2, 0.6, 0.8],
        [0.15, 0.15, 0.85, 0.85],
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_torch_cxcy_to_xywh():
    fixture_boxes_cxcy_torch = torch.from_numpy(fixture_boxes_cxcy)
    result = torch_cxcy_to_xywh(fixture_boxes_cxcy_torch)
    expected = torch.Tensor([
        [0, 0, 10, 10],
        [15, 15, 50, 50],
        [40, 40, 100, 100]
    ])

    assert torch.all(torch.eq(result, expected))

    fixture_boxes_cxcy_torch_norm = torch.from_numpy(fixture_boxes_cxcy_norm)
    result = torch_cxcy_to_xywh(fixture_boxes_cxcy_torch_norm)
    expected = torch.Tensor([
        [0.15, 0.15, 0.3, 0.3],
        [0, 0.2, 0.6, 0.6],
        [0.15, 0.15, 0.7, 0.7],
    ]).type_as(result)

    assert torch.allclose(result, expected)


def test_numpy_backward_conversion():
    xywh = numpy_xyxy_to_xywh(fixture_boxes_xyxy)
    xyxy = numpy_xywh_to_xyxy(xywh)
    np.testing.assert_array_equal(xyxy, fixture_boxes_xyxy)

    cxcy = numpy_xyxy_to_cxcy(fixture_boxes_xyxy)
    xyxy = numpy_cxcy_to_xyxy(cxcy)
    np.testing.assert_array_equal(xyxy, fixture_boxes_xyxy)

    xyxy = numpy_xywh_to_xyxy(fixture_boxes_xywh)
    xywh = numpy_xyxy_to_xywh(xyxy)
    np.testing.assert_array_equal(xywh, fixture_boxes_xywh)

    cxcy = numpy_xywh_to_cxcy(fixture_boxes_xywh)
    xywh = numpy_cxcy_to_xywh(cxcy)
    np.testing.assert_array_equal(xywh, fixture_boxes_xywh)

    xyxy = numpy_cxcy_to_xyxy(fixture_boxes_cxcy)
    cxcy = numpy_xyxy_to_cxcy(xyxy)
    np.testing.assert_array_equal(cxcy, fixture_boxes_cxcy)

    xywh = numpy_cxcy_to_xywh(fixture_boxes_cxcy)
    cxcy = numpy_xywh_to_cxcy(xywh)
    np.testing.assert_array_equal(cxcy, fixture_boxes_cxcy)


def test_torch_backward_conversion():
    xywh = torch_xyxy_to_xywh(torch.from_numpy(fixture_boxes_xyxy))
    xyxy = torch_xywh_to_xyxy(xywh)
    assert torch.all(torch.eq(xyxy, torch.from_numpy(fixture_boxes_xyxy)))

    cxcy = torch_xyxy_to_cxcy(torch.from_numpy(fixture_boxes_xyxy))
    xyxy = torch_cxcy_to_xyxy(cxcy)
    assert torch.all(torch.eq(xyxy, torch.from_numpy(fixture_boxes_xyxy)))

    xyxy = torch_xywh_to_xyxy(torch.from_numpy(fixture_boxes_xywh))
    xywh = torch_xyxy_to_xywh(xyxy)
    assert torch.all(torch.eq(xywh, torch.from_numpy(fixture_boxes_xywh)))

    cxcy = torch_xywh_to_cxcy(torch.from_numpy(fixture_boxes_xywh))
    xywh = torch_cxcy_to_xywh(cxcy)
    assert torch.all(torch.eq(xywh, torch.from_numpy(fixture_boxes_xywh)))

    xyxy = torch_cxcy_to_xyxy(torch.from_numpy(fixture_boxes_cxcy))
    cxcy = torch_xyxy_to_cxcy(xyxy)
    assert torch.all(torch.eq(cxcy, torch.from_numpy(fixture_boxes_cxcy)))

    xywh = torch_cxcy_to_xywh(torch.from_numpy(fixture_boxes_cxcy))
    cxcy = torch_xywh_to_cxcy(xywh)
    assert torch.all(torch.eq(cxcy, torch.from_numpy(fixture_boxes_cxcy))) 