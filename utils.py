# utils.py
import numpy as np
from scipy.interpolate import griddata

def smooth_grid_values(
    current_grid_values, anchor_indices, anchor_values, all_pixel_coords
):
    current_grid_values[anchor_indices] = anchor_values
    anchor_locs = all_pixel_coords[anchor_indices]
    smoothed_grid = griddata(
        anchor_locs, anchor_values, all_pixel_coords, method="cubic"
    )
    mask_nan = np.isnan(smoothed_grid)
    if np.any(mask_nan):
        smoothed_grid[mask_nan] = griddata(
            anchor_locs,
            anchor_values,
            all_pixel_coords[mask_nan],
            method="nearest",
        )
    return smoothed_grid