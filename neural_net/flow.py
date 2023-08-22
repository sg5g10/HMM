import torch
from torch import nn
import numpy as np
from neural_net.maf import sum_except_batch

def merge_leading_dims(x, num_dims):
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)

def split_leading_dim(x, shape):
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)

def repeat_rows(x, num_reps):
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)

class Distribution(nn.Module):
    
    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
 
        inputs = torch.as_tensor(inputs)
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context, batch_size=None):
        context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)
        else:
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

class StandardNormal(Distribution):
    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        neg_energy = -0.5 * \
            sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape,
                                  device=context.device)
            return split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)
            
class Flow(Distribution):
    def __init__(self, transform, distribution, embedding_net=None):
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        self._embedding_net = embedding_net

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=embedded_context)

        noise = merge_leading_dims(noise, num_dims=2)
        embedded_context = repeat_rows(embedded_context, num_reps=num_samples)
        samples, _ = self._transform.inverse(noise, context=embedded_context)

        samples = split_leading_dim(samples, shape=[-1, num_samples])            

        return samples

    