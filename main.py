# main.py
import numpy as np
import pandas as pd
from datetime import datetime

import config
from data_loader import load_data, generate_map_grid, build_total_map, preprocess_data
from solver import run_mcr_als_solver
from analytics import optimize_parameters, generate_predictions_table, generate_summary_statistics

if __name__ == "__main__":
    
    # Update the input file path to match the actual data file
    config.INPUT_FILE = "CRFP_diffdisp.txt"
    
    print("Loading Data...")
    times, stations, anchor_coords, total_matrix, anchor_signals = load_data(config.INPUT_FILE)
    
    # Load and preprocess original data for predictions table
    original_df = pd.read_csv(config.INPUT_FILE, delimiter='\t')  # Use tab delimiter for .txt file
    original_df = preprocess_data(original_df)
    
    print(f"Final dataset: {len(stations)} stations, {len(times)} time steps")

    print("Building Map Grid...")
    all_pixel_coords, Xi, Yi = generate_map_grid(anchor_coords, config.MAP_RESOLUTION)
    global_total_matrix = build_total_map(total_matrix, anchor_coords, stations, all_pixel_coords)

    print("Optimizing Parameters...")
    best_blend, best_ridge = optimize_parameters(
        global_total_matrix, anchor_signals, np.array(stations), anchor_coords, all_pixel_coords
    )

    print("Running Final Solver...")
    all_indices = []
    for s in stations:
        dists = np.sum((all_pixel_coords - np.array(anchor_coords[s])) ** 2, axis=1)
        all_indices.append(np.argmin(dists))

    final_S, final_A = run_mcr_als_solver(
        global_total_matrix, all_indices, anchor_signals, all_pixel_coords, best_blend, best_ridge
    )

    print("Generating Predictions...")
    predictions_df = generate_predictions_table(original_df, final_S, final_A, anchor_coords, all_pixel_coords)
    summary_stats = generate_summary_statistics(predictions_df)
    
    print("Saving Results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = f"{timestamp}_predictions_table.csv"
    summary_file = f"{timestamp}_model_performance_summary.csv"
    
    predictions_df.to_csv(predictions_file, index=False)
    summary_stats.to_csv(summary_file, index=False)
    
    print(f"Files saved: {predictions_file}, {summary_file}")
    
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(summary_stats.to_string(index=False, float_format='%.4f'))
    
    print("\n=== SAMPLE PREDICTIONS ===")
    sample_cols = ['time', 'STATION', 'X', 'Y', 'Layer_1', 'Layer_1_coefficient', 'Layer_1_predicted', 'Layer_1_residual']
    print(predictions_df[sample_cols].head(8).to_string(index=False, float_format='%.4f'))
    
    print(f"\nCompleted! Best params: Blend={best_blend}, Ridge={best_ridge}")
