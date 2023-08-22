import torch
from torch import nn
from neural_net.maf import AffineTransform

class Standardize(nn.Module):
    def __init__(self, mean, std):
        super(Standardize, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return (tensor - self._mean) / self._std


def standardizing_net(
    batch_t,
    min_std: float = 1e-7,
):

    t_mean = torch.mean(batch_t, dim=0)
    t_std = torch.std(batch_t, dim=0)
    t_std[t_std < min_std] = min_std
    nan_in_stats = torch.logical_or(torch.isnan(t_mean).any(), torch.isnan(t_std).any())
    return Standardize(t_mean, t_std)

def standardizing_transform(batch_t, min_std = 1e-14):
    t_mean = torch.mean(batch_t, dim=0)
    t_std = torch.std(batch_t, dim=0)
    t_std[t_std < min_std] = min_std

    return AffineTransform(shift=-t_mean / t_std, scale=1 / t_std)