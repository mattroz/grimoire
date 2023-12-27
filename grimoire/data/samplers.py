import torch

from torch.utils.data import Sampler


class BatchOverfitSampler(Sampler):
    def __init__(self, num_samples: int, batch_size: int):
        """Overfiting Sampler. Mainly for debugging purposes.

        Args:
            num_samples (int): number of sample in the dataset.
            batch_size (int): batch size.
        """
        super().__init__(data_source=None)
        self.batch_size = batch_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return iter(torch.arange(self.batch_size).repeat(self.num_samples))