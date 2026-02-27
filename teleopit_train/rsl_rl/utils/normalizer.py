import numpy as np
import torch

from .utils import reduce_sum, reduce_inplace_sum

class Normalizer(torch.nn.Module):
    def __init__(self, shape, device, init_mean=None, init_std=None, eps=1e-4, clip=np.inf, dtype=torch.float):
        super().__init__()
        
        self._eps = eps
        self._clip = clip
        self.dtype = dtype
        self._build_params(shape, device, init_mean, init_std)
        return

    def record(self, x):
        shape = self.get_shape()
        assert len(x.shape) > len(shape)

        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)

        self._new_count += x.shape[0]
        self._new_sum += torch.sum(x, axis=0)
        self._new_sum_sq += torch.sum(torch.square(x), axis=0)
        
        return

    def update(self):
        if (self._mean_sq is None):
            self._mean_sq = self._calc_mean_sq(self._mean, self._std)
            
        self._new_count = reduce_sum(self._new_count)
        reduce_inplace_sum(self._new_sum)
        reduce_inplace_sum(self._new_sum_sq)
        
        new_count = self._new_count
        new_mean = self._new_sum / new_count
        new_mean_sq = self._new_sum_sq / new_count
        
        new_total = self._count + new_count
        w_old = self._count.type(torch.float) / new_total.type(torch.float)
        w_new = float(new_count) / new_total.type(torch.float)

        self._mean[:] = w_old * self._mean + w_new * new_mean
        self._mean_sq[:] = w_old * self._mean_sq + w_new * new_mean_sq
        self._count[:] = new_total

        self._std[:] = self._calc_std(self._mean, self._mean_sq)

        self._new_count = 0
        self._new_sum[:] = 0
        self._new_sum_sq[:] = 0
        return

    def get_shape(self):
        return self._mean.shape

    def get_count(self):
        return self._count

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def set_mean_std(self, mean, std):
        shape = self.get_shape()
        
        assert mean.shape == shape and std.shape == shape, \
            "Normalizer shape mismatch, expecting size {:d}, but got {:d} and {:d}".format(shape, mean.shape, std.shape)
        
        self._mean[:] = mean
        self._std[:] = std
        self._mean_sq[:] = self._calc_mean_sq(self._mean, self._std)
        return

    def normalize(self, x):
        norm_x = (x - self._mean) / (self._std + self._eps)
        norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        return norm_x.type(self.dtype)

    def unnormalize(self, norm_x):
        x = norm_x * self._std + self._mean
        return x.type(self.dtype)
    
    def _calc_std(self, mean, mean_sq):
        var = mean_sq - torch.square(mean)
        var = torch.clamp_min(var, 1e-8)
        std = torch.sqrt(var)
        std = std.type(self.dtype)
        return std

    def _calc_mean_sq(self, mean, std):
        mean_sq = torch.square(std) + torch.square(mean)
        mean_sq = mean_sq.type(self.dtype)
        return mean_sq

    def _build_params(self, shape, device, init_mean, init_std):
        self._count = torch.nn.Parameter(torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), requires_grad=False)
        self._mean = torch.nn.Parameter(torch.zeros(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)
        self._std = torch.nn.Parameter(torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)

        if init_mean is not None:
            assert init_mean.shape == shape, \
                'Normalizer init mean shape mismatch, expecting {:d}, but got {:d}'.shape(shape, init_mean.shape)
            self._mean[:] = init_mean

        if init_std is not None:
            assert init_std.shape == shape, \
                'Normalizer init std shape mismatch, expecting {:d}, but got {:d}'.format(shape, init_std.shape)
            self._std[:] = init_std

        self._mean_sq = None
        
        self._new_count = 0
        self._new_sum = torch.zeros_like(self._mean)
        self._new_sum_sq = torch.zeros_like(self._mean)
        return