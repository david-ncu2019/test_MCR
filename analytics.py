# analytics.py
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import config
from solver import run_mcr_als_solver # Needs the solver to test parameters

def optimize_parameters(
    total_matrix,
    anchor_signals_dict,
    station_names,
    anchor_coords,
    all_pixel_coords,
):
    print("------------------------------------------------")
    print("STARTING AUTOMATED PARAMETER OPTIMIZATION (CV)")
    print("------------------------------------------------")

    # Indices for KFold
    station_indices = np.arange(len(station_names))
    results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_combinations = len(PARAM_GRID["blending_alpha"]) * len(
        PARAM_GRID["ridge_alpha"]
    )
    count = 0

    for b_alpha in PARAM_GRID["blending_alpha"]:
        for r_alpha in PARAM_GRID["ridge_alpha"]:
            count += 1
            print(
                f"Testing Config {count}/{total_combinations}: [Blend={b_alpha}, Ridge={r_alpha}]...",
                end=" ",
            )

            fold_errors = []

            # --- CROSS VALIDATION LOOP ---
            for train_idx, test_idx in kf.split(station_indices):
                # FIXED HERE: Use train_idx directly on station_names
                train_stations = station_names[train_idx]
                test_stations = station_names[test_idx]

                # Prepare indices for training set
                train_pixel_indices = []
                for s in train_stations:
                    s_loc = anchor_coords[s]
                    dists = np.sum(
                        (all_pixel_coords - np.array(s_loc)) ** 2, axis=1
                    )
                    train_pixel_indices.append(np.argmin(dists))

                # Subset dictionary
                train_signals_dict = {
                    l: anchor_signals_dict[l][train_stations]
                    for l in TARGET_LAYERS
                }

                # Run Solver (Silent Mode)
                S_fold, A_fold = run_mcr_als_solver(
                    total_matrix,
                    train_pixel_indices,
                    train_signals_dict,
                    all_pixel_coords,
                    b_alpha,
                    r_alpha,
                )

                # Validate
                for s_test in test_stations:
                    s_loc = anchor_coords[s_test]
                    dists = np.sum(
                        (all_pixel_coords - np.array(s_loc)) ** 2, axis=1
                    )
                    test_pixel_idx = np.argmin(dists)

                    for k, layer in enumerate(TARGET_LAYERS):
                        pred = A_fold[k, test_pixel_idx] * S_fold[:, k]
                        truth = anchor_signals_dict[layer][s_test].values
                        fold_errors.append(mean_squared_error(truth, pred))

            avg_rmse = np.sqrt(np.mean(fold_errors))
            results.append(
                {
                    "blending_alpha": b_alpha,
                    "ridge_alpha": r_alpha,
                    "rmse": avg_rmse,
                }
            )
            print(f"RMSE: {avg_rmse:.5f}")

    best_config = min(results, key=lambda x: x["rmse"])
    print("------------------------------------------------")
    print(
        f"OPTIMIZATION COMPLETE. Best: Blend={best_config['blending_alpha']}, Ridge={best_config['ridge_alpha']}"
    )
    print("------------------------------------------------")

    return best_config["blending_alpha"], best_config["ridge_alpha"]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def evaluate_prediction_uncertainty(
    total_matrix,
    anchor_signals_dict,
    station_names,
    anchor_coords,
    all_pixel_coords,
    best_blend,
    best_ridge,
    n_iterations=30,
    subsample_ratio=0.8,
):
    """
    Performs Bootstrap Analysis to estimate the uncertainty (Standard Deviation)
    of the spatial maps.

    Args:
        n_iterations: How many times to re-run the model (e.g., 30-50).
        subsample_ratio: Percentage of stations to keep in each run (e.g., 0.8).
    """
    print(f"\nSTARTING UNCERTAINTY ANALYSIS ({n_iterations} runs)...")

    num_pixels = len(all_pixel_coords)
    num_layers = len(TARGET_LAYERS)

    # Storage for all the maps we generate
    # Shape: (Iterations, Layers, Pixels)
    bootstrap_maps = np.zeros((n_iterations, num_layers, num_pixels))

    # 1. Pre-calculate the 'Master' Static Anchor (using ALL data)
    # We use this to ensure signs don't flip between runs (Alignment)
    full_signatures = []
    for layer in TARGET_LAYERS:
        sig = anchor_signals_dict[layer].mean(axis=1).values
        full_signatures.append(sig / np.linalg.norm(sig))
    reference_signatures = np.column_stack(full_signatures)

    # 2. Bootstrap Loop
    for i in range(n_iterations):
        if (i + 1) % 5 == 0:
            print(f"  Bootstrap Run {i+1}/{n_iterations}...")

        # A. Randomly Select a Subset of Stations
        n_samples = int(len(station_names) * subsample_ratio)
        subset_stations = np.random.choice(
            station_names, n_samples, replace=False
        )

        # B. Prepare Indices for this Subset
        subset_indices = []
        for s in subset_stations:
            dists = np.sum(
                (all_pixel_coords - np.array(anchor_coords[s])) ** 2, axis=1
            )
            subset_indices.append(np.argmin(dists))

        # C. Run Solver (Silent Mode)
        # Note: We pass the 'reference_signatures' to keep signs consistent?
        # Actually, the solver builds its own anchors from the subset.
        # We will align them *after* the solver finishes.

        _, A_subset = run_mcr_als_solver(
            total_matrix,
            subset_indices,
            {l: anchor_signals_dict[l][subset_stations] for l in TARGET_LAYERS},
            all_pixel_coords,
            best_blend,
            best_ridge,
        )

        # D. Store Result
        bootstrap_maps[i, :, :] = A_subset

    # 3. Calculate Statistics (Pixel-wise)
    print("Calculating Statistics...")

    # Mean Map (The Robust Prediction)
    mean_map = np.mean(bootstrap_maps, axis=0)

    # Standard Deviation Map (The Uncertainty / Error Bar)
    std_map = np.std(bootstrap_maps, axis=0)

    # Relative Uncertainty (Coefficient of Variation)
    # Avoid division by zero
    cv_map = std_map / (np.abs(mean_map) + 1e-9)

    return mean_map, std_map, cv_map