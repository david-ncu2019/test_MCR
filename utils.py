# utils.py
import numpy as np
from scipy.interpolate import griddata
import config

def smooth_grid_values(
    current_grid_values, anchor_indices, anchor_values, all_pixel_coords
):
    """Smooth grid values with robust NaN handling."""
    # Handle NaN values in inputs
    current_grid_values = np.nan_to_num(current_grid_values, nan=0.0)
    anchor_values = np.nan_to_num(anchor_values, nan=0.0)
    
    # Set anchor points to their true values
    current_grid_values[anchor_indices] = anchor_values
    anchor_locs = all_pixel_coords[anchor_indices]
    
    # Primary interpolation with error handling
    try:
        smoothed_grid = griddata(
            anchor_locs, 
            anchor_values, 
            all_pixel_coords, 
            method=config.INTERPOLATION_METHOD_PRIMARY,
            fill_value=0
        )
    except Exception as e:
        print(f"Warning: Primary interpolation failed ({e}), using fallback method")
        try:
            smoothed_grid = griddata(
                anchor_locs, 
                anchor_values, 
                all_pixel_coords, 
                method='linear',
                fill_value=0
            )
        except Exception as e2:
            print(f"Warning: Linear interpolation failed ({e2}), using nearest neighbor")
            smoothed_grid = griddata(
                anchor_locs, 
                anchor_values, 
                all_pixel_coords, 
                method='nearest',
                fill_value=0
            )
    
    # Handle NaN values with fallback method
    mask_nan = np.isnan(smoothed_grid) | ~np.isfinite(smoothed_grid)
    if np.any(mask_nan):
        try:
            fallback_values = griddata(
                anchor_locs,
                anchor_values,
                all_pixel_coords[mask_nan],
                method=config.INTERPOLATION_METHOD_FALLBACK,
                fill_value=0
            )
            smoothed_grid[mask_nan] = fallback_values
        except:
            # Final fallback - use nearest neighbor
            try:
                fallback_values = griddata(
                    anchor_locs,
                    anchor_values,
                    all_pixel_coords[mask_nan],
                    method='nearest',
                    fill_value=0
                )
                smoothed_grid[mask_nan] = fallback_values
            except:
                # If everything fails, fill with zeros
                smoothed_grid[mask_nan] = 0
    
    # Final cleanup - ensure no NaN or infinite values
    smoothed_grid = np.nan_to_num(smoothed_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    return smoothed_grid
