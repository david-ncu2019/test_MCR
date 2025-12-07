# solver.py
import numpy as np
from sklearn.linear_model import Ridge
import config
from utils import smooth_grid_values # Import the helper function

def run_mcr_als_solver(
    global_total_matrix,
    anchor_indices,
    anchor_signals_dict,
    all_pixel_coords,
    blending_alpha,
    ridge_alpha,
):
    """
    Runs decomposition with automatic stopping criteria.
    """
    num_timesteps, num_pixels = global_total_matrix.shape
    num_layers = len(TARGET_LAYERS)

    # --- Initialize Signatures ---
    initial_signatures = []
    for layer in TARGET_LAYERS:
        avg_sig = anchor_signals_dict[layer].mean(axis=1).values
        avg_sig /= np.linalg.norm(avg_sig)
        initial_signatures.append(avg_sig)

    static_anchor_signatures = np.column_stack(initial_signatures)
    time_signatures = static_anchor_signatures.copy()

    # Initialize Maps
    spatial_maps = np.zeros((num_layers, num_pixels))
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)

    # --- Iteration Loop ---
    MAX_ITER = 100  # High ceiling, but we expect to stop early

    for i in range(MAX_ITER):
        S_prev = time_signatures.copy()  # Store for convergence check

        # 1. Update Maps
        ridge.fit(time_signatures, global_total_matrix)
        spatial_maps = ridge.coef_.T

        # 2. Apply Constraints
        for k, layer_name in enumerate(TARGET_LAYERS):
            true_station_data = anchor_signals_dict[layer_name].values
            current_sig = time_signatures[:, k]
            true_weights = np.dot(true_station_data.T, current_sig)

            spatial_maps[k, :] = smooth_grid_values(
                spatial_maps[k, :],
                anchor_indices,
                true_weights,
                all_pixel_coords,
            )

        # 3. Update Signatures
        ridge.fit(spatial_maps.T, global_total_matrix.T)
        found_signatures = ridge.coef_

        # 4. Blend
        time_signatures = (
            1 - blending_alpha
        ) * static_anchor_signatures + blending_alpha * found_signatures

        # Normalize
        for k in range(num_layers):
            if (
                np.dot(time_signatures[:, k], static_anchor_signatures[:, k])
                < 0
            ):
                time_signatures[:, k] = -time_signatures[:, k]
            time_signatures[:, k] /= np.linalg.norm(time_signatures[:, k])

        # --- 5. STOP CRITERIA CHECK (New) ---
        # Calculate Frobenius norm of the difference (Relative Change)
        diff = np.linalg.norm(time_signatures - S_prev)
        relative_change = diff / (np.linalg.norm(S_prev) + 1e-9)

        if relative_change < CONVERGENCE_TOLERANCE:
            # Uncomment to see convergence details
            # print(f"  Converged at iteration {i+1} (Change: {relative_change:.2e})")
            break

    return time_signatures, spatial_maps
