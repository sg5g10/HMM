import torch
from torch import nn
from torch.nn import functional as F
from neural_net.made import MADE
import numpy as np

def sum_except_batch(x, num_batch_dims=1):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

class InverseNotAvailable(Exception):
    pass

class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()

class CompositeTransform(Transform):
    def __init__(self, transforms):
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)

class AffineTransform(Transform):

    def __init__(
        self, shift= 0.0, scale= 1.0,
    ):        
        super().__init__()
        shift, scale = map(torch.as_tensor, (shift, scale))
        self.register_buffer("_shift", shift)
        self.register_buffer("_scale", scale)   

    @property
    def _log_abs_scale(self):
        return torch.log(torch.abs(self._scale))
    
    def _batch_logabsdet(self, batch_shape):
        if self._log_abs_scale.numel() > 1:
            return self._log_abs_scale.expand(batch_shape).sum()
        else:
            return self._log_abs_scale * torch.Size(batch_shape).numel()

    def forward(self, inputs, context):
        batch_size, *batch_shape = inputs.size()
        outputs = inputs * self._scale + self._shift
        logabsdet = self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs, context):
        batch_size, *batch_shape = inputs.size()
        outputs = (inputs - self._shift) / self._scale
        logabsdet = -self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet

class Permutation(Transform):
    def __init__(self, permutation, dim=1):
        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)

class RandomPermutation(Permutation):
    def __init__(self, features, dim=1):
        super().__init__(torch.randperm(features), dim)
    
class MaskedAutoregressiveTransform(Transform):

    def __init__(self,
                features,
                hidden_features,
                context_features=None,
                num_blocks=2,
                use_residual_blocks=True,
                activation=F.relu,
                dropout_probability=0.0):
        super(MaskedAutoregressiveTransform, self).__init__()
        self.features = features
        autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability
        )
        self._epsilon = 1e-3        
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = int(np.prod(inputs.shape[1:]))
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift   

 



