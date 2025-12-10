# optimize_parameters.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
import config
import data_loader
import solver
import warnings

# Suppress convergence warnings during optimization scans
warnings.filterwarnings('ignore')

class MCROptimizer:
    """
    Scientific parameter optimization for MCR-ALS using K-Fold Cross-Validation.
    Prevents overfitting by testing model performance on unseen stations.
    """
    
    def __init__(self, n_folds=5, random_state=42, verbose=True):
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.results_history = []
    
    def optimize_spatial_parameters(self, param_grid=None):
        """
        Performs Grid Search with Cross-Validation.
        
        Args:
            param_grid (dict): Dictionary of parameters to test.
                               e.g., {'spatial_alpha': [0.1, 0.3], ...}
                               
        Returns:
            best_params (dict): The parameter combination with lowest Error.
            results_df (DataFrame): Table of all test results.
        """
        # 1. Load Data ONCE
        print("[Optimization] Loading Dataset...")
        
        # --- FIX: Unpack 6 values (ignoring the last one) ---
        df, D_df, D, coords, layer_cols, _ = data_loader.load_and_preprocess(config.INPUT_FILE)
        
        station_indices = np.arange(len(D))
        station_names = D_df.index.values
        
        # Default grid if none provided
        if param_grid is None:
            param_grid = {
                'spatial_alpha': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                'anchor_strength': [0.5, 0.7, 0.9, 1.0],
                'spatial_neighbors': [3, 5, 8]
            }
        
        # 2. Iterate through every combination (Grid Search)
        grid = list(ParameterGrid(param_grid))
        print(f"[Optimization] Starting scan of {len(grid)} parameter combinations...")
        
        for i, params in enumerate(grid):
            if self.verbose:
                print(f"  Testing config {i+1}/{len(grid)}: {params}")
            
            fold_errors_mse = []
            fold_errors_mae = []
            
            # 3. K-Fold Cross Validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for train_idx, test_idx in kf.split(station_indices):
                # --- Step A: Prepare Training Data ---
                D_train = D[train_idx]
                coords_train = coords[train_idx]
                
                # Prevent Data Leakage: Generate C_init using ONLY training stations
                train_stations_list = station_names[train_idx]
                df_train = df[df[config.STATION_COL].isin(train_stations_list)]
                
                # Generate initialization/mask for this specific fold
                C_init_train, mask_train = data_loader.generate_ratio_initialization(
                    df_train, layer_cols, train_stations_list
                )
                
                # --- Step B: Run MCR Solver ---
                try:
                    # Pass the current parameters to override config defaults
                    C_train_opt, ST_train_opt = solver.run_mcr_solver(
                        D_train, C_init_train, mask_train, coords_train, 
                        override_params=params
                    )
                except Exception as e:
                    print(f"    ! Solver failed for fold: {e}")
                    continue

                # --- Step C: Predict Test Data (The Validation) ---
                coords_test = coords[test_idx]
                D_test_actual = D[test_idx]
                
                # Interpolate Spatial Maps (C) from Train -> Test locations
                C_test_pred = np.zeros((len(test_idx), C_train_opt.shape[1]))
                
                for k in range(C_train_opt.shape[1]):
                    # Use 'nearest' to handle edge stations (prevents NaNs)
                    C_test_pred[:, k] = griddata(
                        coords_train, 
                        C_train_opt[:, k], 
                        coords_test, 
                        method='nearest' 
                    )
                
                # Reconstruct Data: Space (Interpolated) * Time (Global)
                D_test_pred = np.dot(C_test_pred, ST_train_opt)
                
                # --- Step D: Evaluate Error ---
                mse = mean_squared_error(D_test_actual, D_test_pred)
                mae = mean_absolute_error(D_test_actual, D_test_pred)
                
                fold_errors_mse.append(mse)
                fold_errors_mae.append(mae)
            
            # 4. Aggregate Results
            if fold_errors_mse:
                avg_mse = np.mean(fold_errors_mse)
                avg_mae = np.mean(fold_errors_mae)
                
                result_entry = params.copy()
                result_entry['MSE'] = avg_mse
                result_entry['MAE'] = avg_mae
                self.results_history.append(result_entry)
                
                if self.verbose:
                    print(f"    -> Avg MSE: {avg_mse:.6f} | MAE: {avg_mae:.6f}")
            else:
                if self.verbose:
                    print(f"    -> Failed to compute errors.")

        # 5. Finalize
        results_df = pd.DataFrame(self.results_history)
        
        # Find best parameters (lowest MSE)
        best_row = results_df.loc[results_df['MSE'].idxmin()]
        best_params = best_row.drop(['MSE', 'MAE']).to_dict()
        
        # Convert numeric parameters back to proper types
        if 'spatial_neighbors' in best_params:
            best_params['spatial_neighbors'] = int(best_params['spatial_neighbors'])
            
        print("\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best Parameters found: {best_params}")
        print(f"Lowest MSE: {best_row['MSE']:.6e}")
        
        return best_params, results_df

    def save_results(self, results_df, filename='optimization_results.csv'):
        results_df.to_csv(filename, index=False)
        print(f"Optimization results saved to: {filename}")

# --- Helper Functions for Main ---

def run_quick_optimization():
    """
    Fast scan using ranges defined in config.py
    """
    print("Running Quick Optimization...")
    print(f"Testing Grid: {config.PARAM_GRID_QUICK}")
    
    optimizer = MCROptimizer(n_folds=3, verbose=True)
    
    # Pass the config dictionary
    best_params, results_df = optimizer.optimize_spatial_parameters(config.PARAM_GRID_QUICK)
    
    optimizer.save_results(results_df, 'opt_results_quick.csv')
    return best_params, results_df

def run_full_optimization():
    """
    Deep scan using ranges defined in config.py
    """
    print("Running Full Scientific Optimization...")
    print(f"Testing Grid: {config.PARAM_GRID_FULL}")
    
    optimizer = MCROptimizer(n_folds=5, verbose=True)
    
    # Pass the config dictionary
    best_params, results_df = optimizer.optimize_spatial_parameters(config.PARAM_GRID_FULL)
    
    optimizer.save_results(results_df, 'opt_results_full.csv')
    return best_params, results_df

if __name__ == "__main__":
    # Test run
    run_quick_optimization()