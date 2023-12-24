def assert_shape(length, valid_shape):
    def decorator(func):
        def wrapper(boxes):
            assert len(boxes.shape) == length and boxes.shape[1] == valid_shape, f"Input shape must be [N, {valid_shape}]"
            return func(boxes)
        return wrapper
    return decorator
