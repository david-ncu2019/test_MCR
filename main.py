# main.py
import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTS FROM OUR MODULES ---
import config
from data_loader import load_data, generate_map_grid, build_total_map
from solver import run_mcr_als_solver
from analytics import optimize_parameters, evaluate_prediction_uncertainty

if __name__ == "__main__":

    # 1. Load Data
    print("Loading Data...")
    times, stations, anchor_coords, total_matrix, anchor_signals = load_data(
        config.INPUT_FILE
    )

    # 2. Build Map Grid
    all_pixel_coords, Xi, Yi = generate_map_grid(anchor_coords, config.MAP_RESOLUTION)
    global_total_matrix = build_total_map(
        total_matrix, anchor_coords, stations, all_pixel_coords
    )

    # 3. Optimize Parameters
    # (We convert stations to a numpy array for the optimizer)
    best_blend, best_ridge = optimize_parameters(
        global_total_matrix,
        anchor_signals,
        np.array(stations),
        anchor_coords,
        all_pixel_coords,
    )

    # 4. Run Final Solver (Mean Map)
    print("\nRunning Final Solver...")

    # Prepare indices for all stations
    all_indices = []
    for s in stations:
        dists = np.sum(
            (all_pixel_coords - np.array(anchor_coords[s])) ** 2, axis=1
        )
        all_indices.append(np.argmin(dists))

    final_S, final_A = run_mcr_als_solver(
        global_total_matrix,
        all_indices,
        anchor_signals,
        all_pixel_coords,
        best_blend,
        best_ridge,
    )

    # 5. Run Uncertainty Analysis (Bootstrap)
    mean_A, uncertainty_A, cv_A = evaluate_prediction_uncertainty(
        global_total_matrix,
        anchor_signals,
        np.array(stations),
        anchor_coords,
        all_pixel_coords,
        best_blend,
        best_ridge,
        n_iterations=20,
    )

    print("\nVisualizing Layer 1 Analysis...")

    # 1. Define Station Locations (for plotting)
    station_locs = np.array([anchor_coords[s] for s in stations])

    # 2. Reshape all maps to 2D grids
    # Mean Map (The Prediction)
    coef_grid = mean_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)
    # Standard Deviation (Absolute Error)
    uncert_grid = uncertainty_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)
    # Coefficient of Variation (Relative Error)
    cv_grid = cv_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)

    # 3. Create a 3-Panel Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # --- Plot 1: The Prediction (Mean) ---
    im1 = axes[0].imshow(
        coef_grid,
        extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
        origin="lower",
        cmap="turbo",
    )
    axes[0].set_title(f"1. Prediction (Mean) | Alpha={best_blend}")
    plt.colorbar(im1, ax=axes[0], label="Coefficient Value")

    # --- Plot 2: Absolute Uncertainty (Sigma) ---
    # This is what you saw before (High error at center)
    im2 = axes[1].imshow(
        uncert_grid,
        extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
        origin="lower",
        cmap="inferno",
    )
    axes[1].set_title("2. Absolute Uncertainty (Sigma)")
    plt.colorbar(im2, ax=axes[1], label="Meters (or Units)")
    axes[1].scatter(
        station_locs[:, 0],
        station_locs[:, 1],
        c="white",
        s=5,
        alpha=0.5,
        label="Stations",
    )

    # --- Plot 3: Relative Uncertainty (CV) ---
    # This is the "Truth". It divides Error by Signal.
    # Green = Good (Low %), Red = Bad (High %)
    im3 = axes[2].imshow(
        cv_grid,
        extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
        origin="lower",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.5,
    )
    axes[2].set_title("3. Relative Uncertainty (CV)")
    plt.colorbar(im3, ax=axes[2], label="Relative Error (%)")

    plt.tight_layout()
    plt.show()