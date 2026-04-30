# %% set up
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor
from utils import display_array_as_img

chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

# %% examples
arr = np.load(section_dir / "numbers.npy")

print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch

print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome

arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row

# Exercises - einops operations (match images)
# %% (1) column-stacking
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)
# %% (2) Column-stacking and copying
arr2 = einops.repeat(arr[0], 'c h w -> c (2 h) w')
display_array_as_img(arr2)
# %% (3) Row-stacking and double-copying
arr3 = einops.repeat(arr[:2], 'b c h w -> c (b h) (2 w)')
display_array_as_img(arr3)
# %% (4) Stretching
arr4 = einops.repeat(arr[0], 'c h w -> c (h 2) w')
display_array_as_img(arr4)
# %% (5) Split channels
arr5 = einops.rearrange(arr[0], 'c h w -> h (c w)')
display_array_as_img(arr5)
# %% (6) Stack into rows & cols
arr6 = einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2, b2=3)
display_array_as_img(arr6)
# %% (7) Transpose
arr7 = einops.rearrange(arr[1], 'c h w -> c w h')
display_array_as_img(arr7)
# %% (8) Shrinking
arr8 = einops.reduce(arr, '(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)', 'max', b1=2)
display_array_as_img(arr8)

# Exercises - einops operations & broadcasting
# %% set up
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")
# %% (A1) rearrange
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    return einops.rearrange(t.arange(3, 9), 
                            '(a1 a2) -> a1 a2', a1=3)
    raise NotImplementedError()


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)
# %% (B1) temperature average
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, '(w 7) -> w', 'mean')
    raise NotImplementedError()


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))
# %% (B2) temperature difference
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    return temps - einops.repeat(temperatures_average(temps), 't -> (t 7)')
    raise NotImplementedError()


expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))
# %% (B3) temperature normalized
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    std = einops.reduce(temps, '(w 7) -> w', t.std)
    return temperatures_differences(temps) / einops.repeat(std, 's -> (s 7)')
    raise NotImplementedError()


expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))
# %% (C1) normalize a matrix
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    return matrix / matrix.norm(dim=1, keepdim=True)
    raise NotImplementedError()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)
# %%
