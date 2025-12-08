# robust_mcr_core.py
"""
Robust MCR-ALS Solver Core
Mathematical correctness independent of input data structure.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNorm, ConstraintNonneg
from typing import Tuple, Optional, List, Dict, Any
import warnings

class RobustSpatialConstraint:
    """Spatial smoothing with anchor enforcement."""
    
    def __init__(self, coordinates: np.ndarray, anchor_mask: np.ndarray, 
                 anchor_values: np.ndarray, n_neighbors: int = 5, alpha: float = 0.3):
        """
        Parameters:
        -----------
        coordinates : np.ndarray [n_stations, 2]
            Station coordinates
        anchor_mask : np.ndarray [n_stations, n_components] bool
            True where anchor values exist
        anchor_values : np.ndarray [n_stations, n_components]
            Known coefficient values (only used where anchor_mask=True)
        n_neighbors : int
            Number of spatial neighbors for smoothing
        alpha : float
            Smoothing strength [0,1]. 0=no smoothing, 1=full neighbor average
        """
        self.alpha = alpha
        self.anchor_mask = anchor_mask.copy()
        self.anchor_values = anchor_values.copy()
        
        # Build spatial neighbor network
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(coordinates)
        self.neighbor_indices = self.knn.kneighbors(coordinates, return_distance=False)
    
    def transform(self, C: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing with anchor constraints."""
        C_smooth = C.copy()
        
        # Spatial smoothing
        for i in range(C.shape[0]):
            neighbors = self.neighbor_indices[i]
            neighbor_avg = np.mean(C[neighbors], axis=0)
            C_smooth[i] = (1 - self.alpha) * C[i] + self.alpha * neighbor_avg
        
        # Enforce anchor constraints
        C_smooth[self.anchor_mask] = self.anchor_values[self.anchor_mask]
        
        return C_smooth

class RobustTemporalConstraint:
    """Temporal smoothing with adaptive window sizing."""
    
    def __init__(self, window_length: int = 5, polyorder: int = 2):
        """
        Parameters:
        -----------
        window_length : int
            Savitzky-Golay filter window length
        polyorder : int
            Polynomial order for smoothing
        """
        self.window_length = window_length
        self.polyorder = polyorder
    
    def transform(self, ST: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing with adaptive window sizing."""
        if ST.shape[1] < self.window_length:
            # Fallback for short time series
            return ST
        
        try:
            return savgol_filter(ST, self.window_length, self.polyorder, axis=1)
        except ValueError:
            # Fallback if savgol fails
            warnings.warn("Temporal smoothing failed, returning unsmoothed data")
            return ST

class MCRSolverCore:
    """
    Robust MCR-ALS solver core with mathematical guarantees.
    Input data structure agnostic - users control their own preprocessing.
    """
    
    def __init__(self, ridge_alpha: float = 0.5, max_iterations: int = 50, 
                 convergence_tolerance: float = 1e-6, spatial_neighbors: int = 5,
                 spatial_alpha: float = 0.3, temporal_window: int = 5,
                 temporal_polyorder: int = 2):
        """
        Parameters:
        -----------
        ridge_alpha : float
            Ridge regression regularization strength
        max_iterations : int
            Maximum MCR-ALS iterations
        convergence_tolerance : float
            Convergence threshold for error change
        spatial_neighbors : int
            Number of neighbors for spatial smoothing
        spatial_alpha : float
            Spatial smoothing strength [0,1]
        temporal_window : int
            Temporal smoothing window size
        temporal_polyorder : int
            Polynomial order for temporal smoothing
        """
        self.ridge_alpha = ridge_alpha
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.spatial_neighbors = spatial_neighbors
        self.spatial_alpha = spatial_alpha
        self.temporal_window = temporal_window
        self.temporal_polyorder = temporal_polyorder
        
    def _validate_inputs(self, D: np.ndarray, C_init: np.ndarray, 
                        coordinates: np.ndarray) -> None:
        """Validate input dimensions and data quality."""
        if not isinstance(D, np.ndarray) or D.ndim != 2:
            raise ValueError("D must be 2D numpy array [stations, times]")
        
        if not isinstance(C_init, np.ndarray) or C_init.ndim != 2:
            raise ValueError("C_init must be 2D numpy array [stations, components]")
        
        if not isinstance(coordinates, np.ndarray) or coordinates.shape != (D.shape[0], 2):
            raise ValueError("coordinates must be [stations, 2] array")
        
        if C_init.shape[0] != D.shape[0]:
            raise ValueError(f"Station count mismatch: D has {D.shape[0]}, C_init has {C_init.shape[0]}")
        
        if np.any(np.isnan(D)) or np.any(np.isinf(D)):
            raise ValueError("D contains NaN or infinite values")
        
        if np.any(np.isnan(coordinates)) or np.any(np.isinf(coordinates)):
            raise ValueError("coordinates contain NaN or infinite values")
            
    def _create_anchor_constraints(self, C_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create anchor mask and values from initialization matrix."""
        # Anchor points are non-zero entries in C_init
        anchor_mask = np.abs(C_init) > 1e-9
        anchor_values = C_init.copy()
        
        return anchor_mask, anchor_values
    
    def solve(self, D: np.ndarray, C_init: np.ndarray, coordinates: np.ndarray,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve MCR-ALS decomposition: D ≈ C @ ST
        
        Parameters:
        -----------
        D : np.ndarray [n_stations, n_times]
            Observed data matrix
        C_init : np.ndarray [n_stations, n_components]
            Initial coefficient guess (non-zero values become anchor constraints)
        coordinates : np.ndarray [n_stations, 2]
            Spatial coordinates for smoothing
        verbose : bool
            Print iteration progress
            
        Returns:
        --------
        C_final : np.ndarray [n_stations, n_components]
            Final spatial coefficient matrix
        ST_final : np.ndarray [n_components, n_times]
            Final temporal signature matrix
        diagnostics : dict
            Convergence and error information
        """
        # Validation
        self._validate_inputs(D, C_init, coordinates)
        
        if verbose:
            print(f"MCR-ALS Solver: {D.shape[0]} stations, {D.shape[1]} times, {C_init.shape[1]} components")
        
        # Create constraints
        anchor_mask, anchor_values = self._create_anchor_constraints(C_init)
        n_anchors = np.sum(anchor_mask)
        
        if verbose:
            print(f"Anchor constraints: {n_anchors} total ({n_anchors/anchor_mask.size:.1%} of matrix)")
        
        spatial_constraint = RobustSpatialConstraint(
            coordinates, anchor_mask, anchor_values,
            self.spatial_neighbors, self.spatial_alpha
        )
        
        temporal_constraint = RobustTemporalConstraint(
            self.temporal_window, self.temporal_polyorder
        )
        
        # Setup MCR with robust constraints
        ridge_regressor = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        
        mcr = McrAR(
            c_regr=ridge_regressor,
            st_regr=ridge_regressor,
            c_constraints=[spatial_constraint, ConstraintNonneg()],
            st_constraints=[temporal_constraint, ConstraintNorm(axis=1)],
            max_iter=self.max_iterations,
            tol_err_change=self.convergence_tolerance
        )
        
        # Solve
        try:
            mcr.fit(D, C=C_init, verbose=verbose)
            
            # Collect diagnostics
            diagnostics = {
                'n_iterations': mcr.n_iter,
                'final_error': mcr.err[-1] if mcr.err else None,
                'error_history': mcr.err,
                'convergence_achieved': not mcr.exit_max_iter_reached,
                'exit_condition': {
                    'max_iter': mcr.exit_max_iter_reached,
                    'tol_increase': mcr.exit_tol_increase,
                    'tol_n_increase': mcr.exit_tol_n_increase,
                    'tol_err_change': mcr.exit_tol_err_change
                }
            }
            
            if verbose:
                print(f"Converged in {mcr.n_iter} iterations")
                if diagnostics['final_error']:
                    print(f"Final reconstruction error: {diagnostics['final_error']:.2e}")
            
            return mcr.C_opt_, mcr.ST_opt_, diagnostics
            
        except Exception as e:
            raise RuntimeError(f"MCR-ALS solver failed: {str(e)}")

def load_and_prepare_data(filepath: str, station_col: str, time_col: str, 
                         total_col: str, target_layers: List[str], 
                         coord_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                        pd.DataFrame, pd.DataFrame]:
    """
    Generic data loader - users control their own data structure.
    
    Returns:
    --------
    D : np.ndarray [stations, times]
        Total field observation matrix
    C_init : np.ndarray [stations, components] 
        Initial coefficient matrix (computed from available data)
    coordinates : np.ndarray [stations, 2]
        Station coordinates
    pivot_df : pd.DataFrame
        Pivoted total field data for reference
    metadata_df : pd.DataFrame
        Station metadata
    """
    # Robust file loading
    try:
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) <= 1:
            raise ValueError("Tab separator failed")
    except:
        df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = [station_col, time_col, total_col] + target_layers + coord_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create total field matrix D
    D_df = df.pivot_table(index=station_col, columns=time_col, values=total_col)
    D = D_df.values
    D = np.nan_to_num(D, nan=0.0)  # Handle missing values
    
    # Extract coordinates
    coord_df = df.groupby(station_col)[coord_cols].first()
    coordinates = coord_df.loc[D_df.index].values
    
    # Create initial coefficient matrix from available layer data
    n_stations = len(D_df.index)
    n_components = len(target_layers)
    C_init = np.zeros((n_stations, n_components))
    
    station_to_idx = {station: i for i, station in enumerate(D_df.index)}
    
    # Calculate ratios where data exists
    for j, layer in enumerate(target_layers):
        layer_data = df.dropna(subset=[layer, total_col])
        grouped = layer_data.groupby(station_col)
        
        for station, group in grouped:
            if station in station_to_idx:
                i = station_to_idx[station]
                # Median ratio for robustness
                ratios = group[layer] / (group[total_col] + 1e-9)
                C_init[i, j] = np.clip(np.median(ratios), 0, 1)  # Ensure [0,1] range
    
    print(f"Data loaded: {D.shape[0]} stations, {D.shape[1]} times")
    print(f"Initial coefficients: {np.sum(C_init > 0)} non-zero values")
    
    return D, C_init, coordinates, D_df, coord_df

def reconstruct_predictions(D: np.ndarray, C_final: np.ndarray, ST_final: np.ndarray,
                           station_names: List[str], time_names: List[str], 
                           component_names: List[str]) -> pd.DataFrame:
    """
    Reconstruct component predictions from MCR solution.
    
    Mathematical relationship: Component_ij = C_final[i,:] @ ST_final[:,j]
    """
    n_stations, n_times = D.shape
    n_components = len(component_names)
    
    results = []
    
    for i, station in enumerate(station_names):
        for j, time in enumerate(time_names):
            
            entry = {
                'Station': station,
                'Time': time,
                'Observed_Total': D[i, j]
            }
            
            # Calculate component predictions
            total_predicted = 0
            for k, component in enumerate(component_names):
                pred_value = C_final[i, k] * ST_final[k, j]
                entry[f'{component}_Coeff'] = C_final[i, k]
                entry[f'{component}_Temporal_Sig'] = ST_final[k, j]
                entry[f'{component}_Prediction'] = pred_value
                total_predicted += pred_value
            
            entry['Total_Predicted'] = total_predicted
            entry['Reconstruction_Error'] = abs(D[i, j] - total_predicted)
            
            results.append(entry)
    
    return pd.DataFrame(results)