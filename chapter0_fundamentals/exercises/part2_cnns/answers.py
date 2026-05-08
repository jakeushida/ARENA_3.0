# %% set up
import einops
import tests
import torch as t
import torch.nn as nn
from torch import Tensor


# %% Exercise - implement `ReLU`
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0))
        raise NotImplementedError()


tests.test_relu(ReLU)
# %% Exercise - implement `Linear`
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        bound = 1 / in_features ** 0.5
        weight_mat = t.empty(out_features, in_features).uniform_(-bound, bound)
        bias_vec = t.empty(out_features).uniform_(-bound, bound)

        self.weight = nn.Parameter(weight_mat)
        self.bias = nn.Parameter(bias_vec) if self.use_bias else None
        # raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        out = einops.einsum(self.weight, x, "out_feats in_feats, ... in_feats -> ... out_feats")

        if (self.use_bias):
            out += self.bias

        return out
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias}"
        raise NotImplementedError()


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

linear = Linear(2, 3, True)
print(linear)
# %%
