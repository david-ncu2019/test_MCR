# utils.py
import numpy as np
from scipy.interpolate import griddata
import config

def smooth_grid_values(
    current_grid_values, anchor_indices, anchor_values, all_pixel_coords
):
    # Set anchor points to their true values
    current_grid_values[anchor_indices] = anchor_values
    anchor_locs = all_pixel_coords[anchor_indices]
    
    # Primary interpolation
    smoothed_grid = griddata(
        anchor_locs, 
        anchor_values, 
        all_pixel_coords, 
        method=config.INTERPOLATION_METHOD_PRIMARY
    )
    
    # Handle NaN values with fallback method
    mask_nan = np.isnan(smoothed_grid)
    if np.any(mask_nan):
        smoothed_grid[mask_nan] = griddata(
            anchor_locs,
            anchor_values,
            all_pixel_coords[mask_nan],
            method=config.INTERPOLATION_METHOD_FALLBACK,
        )
    return smoothed_grid
