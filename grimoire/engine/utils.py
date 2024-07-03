import random
import torch

import numpy as numpy


def set_deterministic(seed=42, precision=10):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(precision=precision)

