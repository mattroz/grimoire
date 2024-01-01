import torch
import torch.nn as nn


# fast, but inaccurate (https://github.com/hendrycks/GELUs)
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# slower, than QuickGELU, but more accurate (https://github.com/hendrycks/GELUs)
class FastGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x**2)))