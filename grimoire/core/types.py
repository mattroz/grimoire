import argparse
import torch

import albumentations
import numpy as np
import torch.utils.data as torch_data
from torch.nn.modules.loss import _Loss

from typing import List, Tuple, NamedTuple, Dict

Tensor = torch.Tensor
Device = torch.device
AlbuTransforms = albumentations.Compose
TorchModel = torch.nn.Module
DataLoader = torch_data.DataLoader
Loss = _Loss
TorchOptimizer = torch.optim.Optimizer
ParsedArguments = argparse.Namespace
Bbox = np.ndarray
Label = np.ndarray
Mask = np.ndarray
Image = np.ndarray
TensorImage = torch.Tensor
TensorBbox = torch.Tensor
TensorMask = torch.Tensor
TensorLabel = torch.Tensor