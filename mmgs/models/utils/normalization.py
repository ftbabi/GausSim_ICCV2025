import torch
from torch import nn


class Normalizer(nn.Module):
    """
    Normalizes input data with exponential moving average.
    Highly important for the model's convergence.

    Adopted from MeshGraphNets codebase.
    https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/normalization.py
    """

    def __init__(self, size: int, max_accumulations: float = 10 ** 6, std_epsilon: float = 1e-8,
                 clip_n_sigmas=False, overwrite_mean: float = None, overwrite_std: float = None,
                 force_nonorm: bool = False):
        super().__init__()

        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon)
        self._zero = torch.tensor(0)

        self._acc_count = nn.Parameter(torch.tensor([0.]), requires_grad=False)

        self._num_accumulations = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self._acc_sum = nn.Parameter(torch.zeros(1, size), requires_grad=False)
        self._acc_sum_squared = nn.Parameter(torch.zeros(1, size), requires_grad=False)

        self.clip_n_sigmas = clip_n_sigmas

        self.is_training = False

        self.overwrite_mean = overwrite_mean
        self.overwrite_std = overwrite_std

        self.force_nonorm = force_nonorm

    def inverse(self, normalized_batch_data: torch.FloatTensor, override_mean: float = None) -> torch.FloatTensor:
        """
        Inverse transformation of the normalizer.
        Used to unnormalize output vectors of the model into accelerations

        In MeshGraphNets, statistics for the acceleration normalizer are collected from ground truth data.
        We, instead, collect statistics from sequences generated with linear blend-skinning (see Model.get_positions())

        :param normalized_batch_data: [VxC] data to be unnormalized, should be roughly distributed as N(0, 1)
        :return: [VxC] unnormalized data
        """

        mean = self._mean()
        if override_mean is not None:
            mean = override_mean
        std = self._std_with_epsilon()

        unnormalized_data = normalized_batch_data * std + mean

        return unnormalized_data

    def _accumulate(self, batched_data: torch.FloatTensor):
        """
        Accumulates the batch_data statistics.

        :param batched_data: [VxC] data to accumulate statistics from
        """

        C = batched_data.shape[-1]
        batched_data = batched_data.view(1, -1, C).detach()

        count = batched_data.shape[1]
        data_sum = batched_data.sum(dim=1)
        squared_data_sum = batched_data.pow(2).sum(dim=1)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1.

    def _mean(self) -> torch.FloatTensor:
        """
        Returns the mean of the accumulated data.
        If no data has been accumulated, returns 0.
        :return: [1xC] mean vector
        """
        if self.overwrite_mean is None:
            safe_count = torch.maximum(self._acc_count, torch.tensor(1.).to(self._acc_count))
            return self._acc_sum / safe_count
        else:
            return self.overwrite_mean

    def _std_with_epsilon(self) -> torch.FloatTensor:
        """
        Returns the standard deviation of the accumulated data.
        If no data has been accumulated, returns 1.
        Minimum standard deviation is self._std_epsilon.

        :return: [1xC] standard deviation vector
        """
        if self.overwrite_std is None:
            safe_count = torch.maximum(self._acc_count, torch.tensor(1.).to(self._acc_count))
            std = self._acc_sum_squared / safe_count - self._mean().pow(2)
            std = torch.maximum(std, self._zero.to(std))
            std = std.sqrt()
            std = torch.maximum(std, self._std_epsilon.to(std))
        else:
            std = self.overwrite_std
        return std

    def forward(self, batched_data: torch.FloatTensor, omit_overwrite=False):
        """
        Normalizes the batched_data with exponential moving average.
        If accumulate is True, accumulates the batched_data statistics.

        :param batched_data: [VxC] data to be normalized
        :param accumulate: if True, accumulates the batched_data statistics
            Use True for training and False for testing
        :return: [VxC] normalized data
        """
        # To make sure not grad. should have better choice
        self._acc_count.requires_grad_(False)
        self._num_accumulations.requires_grad_(False)
        self._acc_sum.requires_grad_(False)
        self._acc_sum_squared.requires_grad_(False)
        
        if not self.force_nonorm and self.is_training and self._num_accumulations < self._max_accumulations and not omit_overwrite:
            self._accumulate(batched_data)

        normalized_features = (batched_data - self._mean()) / self._std_with_epsilon()
        return normalized_features

    def train(self, mode: bool = True):
        self.is_training = mode
        return super().train(mode)