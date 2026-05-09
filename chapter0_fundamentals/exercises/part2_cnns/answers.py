# %% set up
import sys
from pathlib import Path

import einops
import torch as t
import torch.nn as nn
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part2_cnns.tests as tests


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
# %% flatten
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
# %% Exercise - implement the simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28 ** 2, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)
        # raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))
        raise NotImplementedError()


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)
# %%
