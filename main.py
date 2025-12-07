# main.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- IMPORTS FROM OUR MODULES ---
import config
from data_loader import load_data, generate_map_grid, build_total_map
from solver import run_mcr_als_solver
from analytics import (
    optimize_parameters, 
    evaluate_prediction_uncertainty,
    generate_predictions_table,
    generate_summary_statistics
)

if __name__ == "__main__":

    # 1. Load Data
    print("Loading Data...")
    times, stations, anchor_coords, total_matrix, anchor_signals = load_data(
        config.INPUT_FILE
    )
    
    # Load original DataFrame for table generation
    original_df = pd.read_csv(config.INPUT_FILE)

    # 2. Build Map Grid
    all_pixel_coords, Xi, Yi = generate_map_grid(anchor_coords, config.MAP_RESOLUTION)
    global_total_matrix = build_total_map(
        total_matrix, anchor_coords, stations, all_pixel_coords
    )

    # 3. Optimize Parameters
    best_blend, best_ridge = optimize_parameters(
        global_total_matrix,
        anchor_signals,
        np.array(stations),
        anchor_coords,
        all_pixel_coords,
    )

    # 4. Run Final Solver
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

    # 5. Generate Predictions Table
    print("\nGenerating Predictions Table...")
    predictions_df = generate_predictions_table(
        original_df,
        final_S,
        final_A,
        anchor_coords,
        all_pixel_coords,
    )
    
    # 6. Generate Summary Statistics
    summary_stats = generate_summary_statistics(predictions_df)
    
    # 7. Save Results
    print("Saving results...")
    predictions_df.to_csv('predictions_table.csv', index=False)
    summary_stats.to_csv('model_performance_summary.csv', index=False)
    
    print("\nResults saved:")
    print("- predictions_table.csv: Full table with original and predicted values")
    print("- model_performance_summary.csv: Summary statistics by layer")
    
    # Display summary
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(summary_stats.to_string(index=False, float_format='%.4f'))
    
    # Show sample of predictions table
    print("\n=== SAMPLE OF PREDICTIONS TABLE ===")
    sample_cols = ['time', 'STATION', 'X', 'Y', 'Layer_1', 'Layer_1_predicted', 'Layer_1_residual']
    print(predictions_df[sample_cols].head(10).to_string(index=False, float_format='%.4f'))

    # 8. Optional: Run Uncertainty Analysis (commented out for speed)
    # mean_A, uncertainty_A, cv_A = evaluate_prediction_uncertainty(
    #     global_total_matrix,
    #     anchor_signals,
    #     np.array(stations),
    #     anchor_coords,
    #     all_pixel_coords,
    #     best_blend,
    #     best_ridge,
    # )

    # 9. Optional: Visualization (commented out but preserved)
    # print("\nVisualizing Layer 1 Analysis...")
    # station_locs = np.array([anchor_coords[s] for s in stations])
    # coef_grid = mean_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)
    # uncert_grid = uncertainty_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)
    # cv_grid = cv_A[0, :].reshape(config.MAP_RESOLUTION, config.MAP_RESOLUTION)
    # 
    # fig, axes = plt.subplots(1, 3, figsize=config.FIGURE_SIZE)
    # 
    # im1 = axes[0].imshow(
    #     coef_grid,
    #     extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
    #     origin="lower",
    #     cmap=config.COLORMAP_PREDICTION,
    # )
    # axes[0].set_title(f"1. Prediction (Mean) | Alpha={best_blend}")
    # plt.colorbar(im1, ax=axes[0], label="Coefficient Value")
    # 
    # im2 = axes[1].imshow(
    #     uncert_grid,
    #     extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
    #     origin="lower",
    #     cmap=config.COLORMAP_UNCERTAINTY,
    # )
    # axes[1].set_title("2. Absolute Uncertainty (Sigma)")
    # plt.colorbar(im2, ax=axes[1], label="Meters (or Units)")
    # axes[1].scatter(
    #     station_locs[:, 0],
    #     station_locs[:, 1],
    #     c=config.STATION_SCATTER_CONFIG['color'],
    #     s=config.STATION_SCATTER_CONFIG['size'],
    #     alpha=config.STATION_SCATTER_CONFIG['alpha'],
    #     label=config.STATION_SCATTER_CONFIG['label'],
    # )
    # 
    # im3 = axes[2].imshow(
    #     cv_grid,
    #     extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
    #     origin="lower",
    #     cmap=config.COLORMAP_RELATIVE,
    #     vmin=config.CV_PLOT_VMIN,
    #     vmax=config.CV_PLOT_VMAX,
    # )
    # axes[2].set_title("3. Relative Uncertainty (CV)")
    # plt.colorbar(im3, ax=axes[2], label="Relative Error (%)")
    # 
    # plt.tight_layout()
    # plt.show()