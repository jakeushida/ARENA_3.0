# %% set up
from pathlib import Path

import einops
import numpy as np
from part0_prereqs.utils import display_array_as_img

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
# %%
