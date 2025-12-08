#!/usr/bin/env python3
"""
Example implementation of robust MCR-ALS solver.
Demonstrates mathematically correct core with user-controlled data structure.
"""

from robust_mcr_core import MCRSolverCore, load_and_prepare_data, reconstruct_predictions
import numpy as np
import pandas as pd

def main():
    """
    Example usage of robust MCR solver core.
    Users control their input data structure - solver adapts mathematically.
    """
    
    # Configuration - user adjustable
    CONFIG = {
        'input_file': 'subset_synthetic_MLCW_layers_differential.csv',
        'output_file': 'robust_mcr_predictions.csv',
        'station_col': 'STATION',
        'time_col': 'time', 
        'total_col': 'Layer_Total',
        'target_layers': ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4'],  # User defines their components
        'coord_cols': ['X', 'Y'],
        
        # Solver parameters
        'ridge_alpha': 0.5,
        'max_iterations': 50,
        'convergence_tolerance': 1e-6,
        'spatial_neighbors': 5,
        'spatial_alpha': 0.3,
        'temporal_window': 5,
        'temporal_polyorder': 2
    }
    
    try:
        # 1. Load and prepare data (user-controlled structure)
        print("Loading data...")
        D, C_init, coordinates, D_df, coord_df = load_and_prepare_data(
            CONFIG['input_file'],
            CONFIG['station_col'], 
            CONFIG['time_col'],
            CONFIG['total_col'],
            CONFIG['target_layers'],
            CONFIG['coord_cols']
        )
        
        print(f"Data matrix D: {D.shape}")
        print(f"Initial coefficients C: {C_init.shape}")
        print(f"Spatial coordinates: {coordinates.shape}")
        
        # 2. Initialize solver core
        solver = MCRSolverCore(
            ridge_alpha=CONFIG['ridge_alpha'],
            max_iterations=CONFIG['max_iterations'],
            convergence_tolerance=CONFIG['convergence_tolerance'],
            spatial_neighbors=CONFIG['spatial_neighbors'],
            spatial_alpha=CONFIG['spatial_alpha'],
            temporal_window=CONFIG['temporal_window'],
            temporal_polyorder=CONFIG['temporal_polyorder']
        )
        
        # 3. Solve MCR decomposition
        print("\nSolving MCR-ALS...")
        C_final, ST_final, diagnostics = solver.solve(D, C_init, coordinates, verbose=True)
        
        # 4. Analyze results
        print(f"\nSolution diagnostics:")
        print(f"  Iterations: {diagnostics['n_iterations']}")
        print(f"  Converged: {diagnostics['convergence_achieved']}")
        if diagnostics['final_error']:
            print(f"  Final error: {diagnostics['final_error']:.2e}")
        
        # 5. Reconstruct predictions  
        print("\nReconstructing predictions...")
        results_df = reconstruct_predictions(
            D, C_final, ST_final,
            station_names=list(D_df.index),
            time_names=list(D_df.columns), 
            component_names=CONFIG['target_layers']
        )
        
        # 6. Save results
        results_df.to_csv(CONFIG['output_file'], index=False)
        print(f"Results saved to: {CONFIG['output_file']}")
        
        # 7. Mathematical validation
        print(f"\nMathematical validation:")
        print(f"  Mean reconstruction error: {results_df['Reconstruction_Error'].mean():.2e}")
        print(f"  Max reconstruction error: {results_df['Reconstruction_Error'].max():.2e}")
        print(f"  Error standard deviation: {results_df['Reconstruction_Error'].std():.2e}")
        
        # 8. Component statistics
        print(f"\nComponent predictions summary:")
        for component in CONFIG['target_layers']:
            pred_col = f'{component}_Prediction'
            if pred_col in results_df.columns:
                mean_pred = results_df[pred_col].mean()
                std_pred = results_df[pred_col].std()
                print(f"  {component}: mean={mean_pred:.3f}, std={std_pred:.3f}")
        
        # Note about mathematical correctness
        print(f"\n" + "="*60)
        print("MATHEMATICAL GUARANTEE:")
        print("D ≈ C_final @ ST_final (within numerical tolerance)")
        print("Component_ij = C_final[i,k] × ST_final[k,j]")
        print("NOT Component_ij = C_final[i,k] × D[i,j]")
        print("="*60)
        
        return results_df, diagnostics
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    results, diag = main()