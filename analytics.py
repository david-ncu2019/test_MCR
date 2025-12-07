# analytics.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import config
from solver import run_mcr_als_solver

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

    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_RANDOM_STATE)

    total_combinations = len(config.PARAM_GRID["blending_alpha"]) * len(
        config.PARAM_GRID["ridge_alpha"]
    )
    count = 0

    for b_alpha in config.PARAM_GRID["blending_alpha"]:
        for r_alpha in config.PARAM_GRID["ridge_alpha"]:
            count += 1
            print(
                f"Testing Config {count}/{total_combinations}: [Blend={b_alpha}, Ridge={r_alpha}]...",
                end=" ",
            )

            fold_errors = []

            # --- CROSS VALIDATION LOOP ---
            for train_idx, test_idx in kf.split(station_indices):
                # Use train_idx directly on station_names
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

                # Subset dictionary with NaN handling
                train_signals_dict = {}
                for l in config.TARGET_LAYERS:
                    subset_data = anchor_signals_dict[l][train_stations]
                    # Ensure no NaN values in training data
                    subset_data = subset_data.fillna(0)
                    train_signals_dict[l] = subset_data

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

                    for k, layer in enumerate(config.TARGET_LAYERS):
                        pred = A_fold[k, test_pixel_idx] * S_fold[:, k]
                        truth = anchor_signals_dict[layer][s_test].values
                        
                        # Handle NaN values in truth data
                        truth = np.nan_to_num(truth, nan=0.0)
                        pred = np.nan_to_num(pred, nan=0.0)
                        
                        # Only compute MSE if we have valid data
                        if len(truth) > 0 and len(pred) > 0:
                            fold_errors.append(mean_squared_error(truth, pred))

            if len(fold_errors) > 0:
                avg_rmse = np.sqrt(np.mean(fold_errors))
            else:
                avg_rmse = 1e6  # Large error if no valid data
                
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


def evaluate_prediction_uncertainty(
    total_matrix,
    anchor_signals_dict,
    station_names,
    anchor_coords,
    all_pixel_coords,
    best_blend,
    best_ridge,
    n_iterations=None,
    subsample_ratio=None,
):
    """
    Performs Bootstrap Analysis to estimate the uncertainty (Standard Deviation)
    of the spatial maps.

    Args:
        n_iterations: How many times to re-run the model. Uses config default if None.
        subsample_ratio: Percentage of stations to keep in each run. Uses config default if None.
    """
    # Use config defaults if parameters not specified
    if n_iterations is None:
        n_iterations = config.BOOTSTRAP_ITERATIONS
    if subsample_ratio is None:
        subsample_ratio = config.SUBSAMPLE_RATIO
        
    print(f"\nSTARTING UNCERTAINTY ANALYSIS ({n_iterations} runs)...")

    num_pixels = len(all_pixel_coords)
    num_layers = len(config.TARGET_LAYERS)

    # Storage for all the maps we generate
    # Shape: (Iterations, Layers, Pixels)
    bootstrap_maps = np.zeros((n_iterations, num_layers, num_pixels))

    # 1. Pre-calculate the 'Master' Static Anchor (using ALL data)
    # We use this to ensure signs don't flip between runs (Alignment)
    full_signatures = []
    for layer in config.TARGET_LAYERS:
        sig = anchor_signals_dict[layer].mean(axis=1).values
        sig = np.nan_to_num(sig, nan=0.0)  # Handle NaN
        sig_norm = np.linalg.norm(sig)
        if sig_norm > 0:
            sig = sig / sig_norm
        full_signatures.append(sig)
    reference_signatures = np.column_stack(full_signatures)

    # 2. Bootstrap Loop
    for i in range(n_iterations):
        if (i + 1) % config.BOOTSTRAP_PRINT_INTERVAL == 0:
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

        # C. Prepare subset signals with NaN handling
        subset_signals = {}
        for l in config.TARGET_LAYERS:
            subset_data = anchor_signals_dict[l][subset_stations]
            subset_data = subset_data.fillna(0)  # Handle NaN
            subset_signals[l] = subset_data

        # C. Run Solver (Silent Mode)
        _, A_subset = run_mcr_als_solver(
            total_matrix,
            subset_indices,
            subset_signals,
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
    cv_map = std_map / (np.abs(mean_map) + config.EPSILON)

    return mean_map, std_map, cv_map


def generate_predictions_table(
    original_df,
    time_signatures,
    spatial_maps,
    anchor_coords,
    all_pixel_coords,
):
    """
    Generate a comprehensive table with original data, coefficients, and predictions.
    
    Args:
        original_df: Original DataFrame with all data
        time_signatures: S matrix from MCR-ALS (time x layers)
        spatial_maps: A matrix from MCR-ALS (layers x pixels)
        anchor_coords: Dictionary mapping station names to coordinates
        all_pixel_coords: Array of all pixel coordinates
    
    Returns:
        pandas.DataFrame: Table with original columns plus coefficients and predictions
    """
    print("Generating predictions table...")
    
    # Create a copy of original data to avoid modifying input
    results_df = original_df.copy()
    
    # Find pixel indices for each station
    station_pixel_map = {}
    for station in anchor_coords.keys():
        s_loc = np.array(anchor_coords[station])
        dists = np.sum((all_pixel_coords - s_loc) ** 2, axis=1)
        station_pixel_map[station] = np.argmin(dists)
    
    # Add coefficient columns for each layer
    for layer_idx, layer_name in enumerate(config.TARGET_LAYERS):
        coef_column_name = f"{layer_name}_coefficient"
        results_df[coef_column_name] = np.nan
        
        # For each row, add the spatial coefficient
        for idx, row in results_df.iterrows():
            station = row['STATION']
            pixel_idx = station_pixel_map[station]
            results_df.loc[idx, coef_column_name] = spatial_maps[layer_idx, pixel_idx]
    
    # Generate predictions for each layer
    for layer_idx, layer_name in enumerate(config.TARGET_LAYERS):
        pred_column_name = f"{layer_name}_predicted"
        results_df[pred_column_name] = np.nan
        
        # For each row in the original data
        for idx, row in results_df.iterrows():
            station = row['STATION']
            time_idx = np.where(original_df['time'] == row['time'])[0][0]
            
            # Get pixel index for this station
            pixel_idx = station_pixel_map[station]
            
            # Calculate prediction: A[layer, pixel] * S[time, layer]
            prediction = spatial_maps[layer_idx, pixel_idx] * time_signatures[time_idx, layer_idx]
            results_df.loc[idx, pred_column_name] = prediction
    
    # Generate predicted Layer_Total
    results_df['Layer_Total_predicted'] = np.nan
    for idx, row in results_df.iterrows():
        station = row['STATION']
        time_idx = np.where(original_df['time'] == row['time'])[0][0]
        pixel_idx = station_pixel_map[station]
        
        # Sum all layer contributions for total
        total_pred = sum(
            spatial_maps[k, pixel_idx] * time_signatures[time_idx, k] 
            for k in range(len(config.TARGET_LAYERS))
        )
        results_df.loc[idx, 'Layer_Total_predicted'] = total_pred
    
    # Calculate residuals (differences)
    for layer_name in config.TARGET_LAYERS:
        results_df[f"{layer_name}_residual"] = (
            results_df[layer_name] - results_df[f"{layer_name}_predicted"]
        )
    
    results_df['Layer_Total_residual'] = (
        results_df['Layer_Total'] - results_df['Layer_Total_predicted']
    )
    
    return results_df


def generate_summary_statistics(predictions_df):
    """
    Generate summary statistics for model performance.
    
    Args:
        predictions_df: DataFrame from generate_predictions_table
    
    Returns:
        pandas.DataFrame: Summary statistics by layer
    """
    print("Calculating summary statistics...")
    
    summary_stats = []
    
    # Statistics for each layer
    for layer_name in config.TARGET_LAYERS + ['Layer_Total']:
        original_col = layer_name
        predicted_col = f"{layer_name}_predicted"
        residual_col = f"{layer_name}_residual"
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions_df[residual_col]))
        rmse = np.sqrt(np.mean(predictions_df[residual_col] ** 2))
        r2 = np.corrcoef(predictions_df[original_col], predictions_df[predicted_col])[0, 1] ** 2
        
        summary_stats.append({
            'Layer': layer_name,
            'MAE': mae,
            'RMSE': rmse,
            'R_sq': r2,
            'Mean_Original': predictions_df[original_col].mean(),
            'Mean_Predicted': predictions_df[predicted_col].mean(),
            'Std_Original': predictions_df[original_col].std(),
            'Std_Predicted': predictions_df[predicted_col].std()
        })
    
    return pd.DataFrame(summary_stats)
