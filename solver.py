# solver.py
import numpy as np
from sklearn.linear_model import Ridge
import config
from utils import smooth_grid_values

def run_mcr_als_solver(
    global_total_matrix,
    anchor_indices,
    anchor_signals_dict,
    all_pixel_coords,
    blending_alpha,
    ridge_alpha,
):
    """
    Runs decomposition with automatic stopping criteria and robust NaN handling.
    """
    num_timesteps, num_pixels = global_total_matrix.shape
    num_layers = len(config.TARGET_LAYERS)

    # --- Initialize Signatures ---
    initial_signatures = []
    for layer in config.TARGET_LAYERS:
        avg_sig = anchor_signals_dict[layer].mean(axis=1).values
        # Handle NaN in signatures
        avg_sig = np.nan_to_num(avg_sig, nan=0.0)
        sig_norm = np.linalg.norm(avg_sig)
        if sig_norm > 0:
            avg_sig /= sig_norm
        initial_signatures.append(avg_sig)

    static_anchor_signatures = np.column_stack(initial_signatures)
    time_signatures = static_anchor_signatures.copy()

    # Initialize Maps
    spatial_maps = np.zeros((num_layers, num_pixels))
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=config.RIDGE_FIT_INTERCEPT)

    # --- Iteration Loop ---
    for i in range(config.MAX_ITERATIONS):
        S_prev = time_signatures.copy()  # Store for convergence check

        # 1. Update Maps with NaN handling
        # Clean input data for Ridge regression
        X_clean = np.nan_to_num(time_signatures, nan=0.0)
        y_clean = np.nan_to_num(global_total_matrix, nan=0.0)
        
        # Check for valid input
        if np.any(np.isnan(X_clean)) or np.any(np.isnan(y_clean)):
            print("Warning: NaN values detected and replaced with zeros")
            X_clean = np.nan_to_num(X_clean, nan=0.0)
            y_clean = np.nan_to_num(y_clean, nan=0.0)
        
        ridge.fit(X_clean, y_clean)
        spatial_maps = ridge.coef_.T
        spatial_maps = np.nan_to_num(spatial_maps, nan=0.0)

        # 2. Apply Constraints
        for k, layer_name in enumerate(config.TARGET_LAYERS):
            anchor_data = anchor_signals_dict[layer_name].values
            # Handle NaN in anchor data
            anchor_data = np.nan_to_num(anchor_data, nan=0.0)
            
            current_sig = time_signatures[:, k]
            true_weights = np.dot(anchor_data.T, current_sig)
            # Handle NaN in weights
            true_weights = np.nan_to_num(true_weights, nan=0.0)

            spatial_maps[k, :] = smooth_grid_values(
                spatial_maps[k, :],
                anchor_indices,
                true_weights,
                all_pixel_coords,
            )

        # 3. Update Signatures with NaN handling
        spatial_maps_clean = np.nan_to_num(spatial_maps.T, nan=0.0)
        y_T_clean = np.nan_to_num(global_total_matrix.T, nan=0.0)
        
        ridge.fit(spatial_maps_clean, y_T_clean)
        found_signatures = ridge.coef_
        found_signatures = np.nan_to_num(found_signatures, nan=0.0)

        # 4. Blend
        time_signatures = (
            1 - blending_alpha
        ) * static_anchor_signatures + blending_alpha * found_signatures

        # Normalize with NaN protection
        for k in range(num_layers):
            if (
                np.dot(time_signatures[:, k], static_anchor_signatures[:, k])
                < 0
            ):
                time_signatures[:, k] = -time_signatures[:, k]
            
            sig_norm = np.linalg.norm(time_signatures[:, k])
            if sig_norm > 0:
                time_signatures[:, k] /= sig_norm
            
            # Final NaN check
            time_signatures[:, k] = np.nan_to_num(time_signatures[:, k], nan=0.0)

        # --- 5. STOP CRITERIA CHECK ---
        # Calculate Frobenius norm of the difference (Relative Change)
        diff = np.linalg.norm(time_signatures - S_prev)
        relative_change = diff / (np.linalg.norm(S_prev) + config.EPSILON)

        if relative_change < config.CONVERGENCE_TOLERANCE:
            break

    # Final cleanup of outputs
    time_signatures = np.nan_to_num(time_signatures, nan=0.0)
    spatial_maps = np.nan_to_num(spatial_maps, nan=0.0)

    return time_signatures, spatial_maps
