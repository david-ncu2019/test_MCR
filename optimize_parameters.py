# optimize_parameters.py - Scientific parameter optimization via cross-validation
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
import config
import data_loader
import solver
import warnings
warnings.filterwarnings('ignore')

class MCROptimizer:
    """Cross-validation optimization for MCR-ALS hyperparameters."""
    
    def __init__(self, n_folds=5, random_state=42, verbose=True):
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.results_history = []
    
    def optimize_spatial_parameters(self, param_grid=None):
        """
        Optimize spatial parameters using K-fold cross-validation.
        
        Returns:
        --------
        best_params : dict
            Optimal parameter values
        results_df : pd.DataFrame
            Full optimization results
        """
        if param_grid is None:
            param_grid = {
                'spatial_alpha': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                'anchor_strength': [0.3, 0.5, 0.7, 0.9],
                'spatial_neighbors': [3, 5, 8]
            }
        
        print("--- MCR-ALS Hyperparameter Optimization ---")
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation: {self.n_folds}-fold")
        
        # Load data once
        df, D_df, D, coords, layer_cols = data_loader.load_and_preprocess(config.INPUT_FILE)
        station_indices = np.arange(len(D))
        
        results = []
        param_combinations = list(ParameterGrid(param_grid))
        
        for i, params in enumerate(param_combinations):
            if self.verbose:
                print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
            
            fold_scores = self._cross_validate_params(
                params, df, D_df, D, coords, layer_cols, station_indices
            )
            
            if fold_scores:
                avg_mse = np.mean(fold_scores)
                std_mse = np.std(fold_scores)
                
                result = {
                    **params,
                    'mean_mse': avg_mse,
                    'std_mse': std_mse,
                    'fold_scores': fold_scores
                }
                results.append(result)
                
                if self.verbose:
                    print(f"  -> MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        
        # Find best parameters
        if not results:
            raise RuntimeError("No valid parameter combinations found")
        
        results_df = pd.DataFrame(results)
        best_idx = results_df['mean_mse'].idxmin()
        best_params = results_df.iloc[best_idx].to_dict()
        
        print(f"\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best parameters: {best_params}")
        print(f"Best MSE: {best_params['mean_mse']:.6f}")
        
        self.results_history.append(results_df)
        return best_params, results_df
    
    def _cross_validate_params(self, params, df, D_df, D, coords, layer_cols, station_indices):
        """Run cross-validation for given parameters."""
        fold_scores = []
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(station_indices):
            try:
                score = self._evaluate_single_fold(
                    params, df, D_df, D, coords, layer_cols, train_idx, test_idx
                )
                if score is not None:
                    fold_scores.append(score)
            except Exception as e:
                if self.verbose:
                    print(f"    Fold failed: {str(e)}")
                continue
        
        return fold_scores if len(fold_scores) >= self.n_folds // 2 else None
    
    def _evaluate_single_fold(self, params, df, D_df, D, coords, layer_cols, train_idx, test_idx):
        """Evaluate single cross-validation fold."""
        # Prepare training data
        D_train = D[train_idx]
        coords_train = coords[train_idx]
        train_stations = D_df.index[train_idx]
        
        df_train = df[df[config.STATION_COL].isin(train_stations)]
        C_init_train, mask_train = data_loader.generate_ratio_initialization(
            df_train, layer_cols, train_stations
        )
        
        # Run MCR-ALS on training set
        C_train, ST_train = solver.run_mcr_solver(
            D_train, C_init_train, mask_train, coords_train, override_params=params
        )
        
        # Predict test set
        coords_test = coords[test_idx]
        D_test_actual = D[test_idx]
        
        # Interpolate coefficients to test locations
        C_test_pred = self._interpolate_coefficients(coords_train, C_train, coords_test)
        
        # Reconstruct test data
        D_test_pred = np.dot(C_test_pred, ST_train)
        
        # Calculate prediction error
        valid_mask = (D_test_actual != 0) & ~np.isnan(D_test_actual)
        if np.sum(valid_mask) < 5:  # Need minimum valid points
            return None
        
        mse = mean_squared_error(D_test_actual[valid_mask], D_test_pred[valid_mask])
        return mse
    
    def _interpolate_coefficients(self, coords_train, C_train, coords_test):
        """Interpolate spatial coefficients to test locations."""
        C_test_pred = np.zeros((len(coords_test), C_train.shape[1]))
        
        for k in range(C_train.shape[1]):
            # Try different interpolation methods as fallback
            methods = ['linear', 'nearest']
            for method in methods:
                try:
                    C_test_pred[:, k] = griddata(
                        coords_train, C_train[:, k], coords_test, 
                        method=method, fill_value=0
                    )
                    break
                except:
                    continue
        
        return C_test_pred
    
    def save_results(self, results_df, filename='optimization_results.csv'):
        """Save optimization results to CSV."""
        # Flatten fold_scores for saving
        results_save = results_df.copy()
        results_save['fold_scores_str'] = results_save['fold_scores'].astype(str)
        results_save = results_save.drop('fold_scores', axis=1)
        
        results_save.to_csv(filename, index=False)
        print(f"Optimization results saved to: {filename}")

def run_quick_optimization():
    """Quick optimization with smaller parameter grid."""
    param_grid = {
        'spatial_alpha': [0.1, 0.3, 0.5, 0.7],
        'anchor_strength': [0.5, 0.7],
        'spatial_neighbors': [5, 8]
    }
    
    optimizer = MCROptimizer(n_folds=3, verbose=True)
    best_params, results_df = optimizer.optimize_spatial_parameters(param_grid)
    optimizer.save_results(results_df)
    
    return best_params, results_df

def run_full_optimization():
    """Comprehensive optimization with full parameter grid."""
    param_grid = {
        'spatial_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'anchor_strength': [0.3, 0.5, 0.7, 0.9],
        'spatial_neighbors': [3, 5, 8, 10],
    }
    
    optimizer = MCROptimizer(n_folds=5, verbose=True)
    best_params, results_df = optimizer.optimize_spatial_parameters(param_grid)
    optimizer.save_results(results_df)
    
    return best_params, results_df

if __name__ == "__main__":
    # Run quick optimization by default
    print("Running quick optimization...")
    best_params, results = run_quick_optimization()
    
    print(f"\nTo use optimized parameters, update config.py:")
    print(f"SPATIAL_ALPHA = {best_params['spatial_alpha']}")
    print(f"ANCHOR_STRENGTH = {best_params['anchor_strength']}")
    print(f"SPATIAL_NEIGHBORS = {best_params['spatial_neighbors']}")
