# main.py
import config
import data_loader
import solver
import analytics
import os
import json

def run_optimization():
    """Run parameter optimization if requested."""
    from optimize_parameters import run_quick_optimization
    
    if input("Run parameter optimization? (y/n): ").lower() == 'y':
        print("Running optimization...")
        best_params, _ = run_quick_optimization()
        
        with open('optimal_parameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        return best_params
    return None

def main():
    print(f"--- Starting MCR Workflow with Coordinates ---")

    # Check for optimization
    optimized_params = None
    if os.path.exists('optimal_parameters.json'):
        with open('optimal_parameters.json', 'r') as f:
            optimized_params = json.load(f)
        print(f"Using optimized parameters: {optimized_params}")
    else:
        optimized_params = run_optimization()
    
    # 1. Load Data (now returns coordinate dataframe)
    print("[1/4] Loading Data...")
    df, D_df, D, coords, layer_cols, coord_df = data_loader.load_and_preprocess(config.INPUT_FILE)
    print(f"Loaded {len(coord_df)} stations with coordinates")
    
    # 2. Init with Mask
    print("[2/4] Generating Sparse Initialization...")
    C_init, anchor_mask = data_loader.generate_ratio_initialization(df, layer_cols, D_df.index)
    
    # 3. Solve with Mask
    print("[3/4] Running MCR-ALS Solver...")
    C_final, ST_final = solver.run_mcr_solver(D, C_init, anchor_mask, coords, optimized_params)
    
    # 4. Save with Coordinates
    print("[4/4] Generating Outputs with Coordinates...")
    analytics.save_predictions(
        df, D_df, C_final, ST_final, layer_cols, config.OUTPUT_FILE, coord_df
    )
    
    print("\nWorkflow Completed Successfully.")
    print("Output file now includes station coordinates for GIS integration.")

if __name__ == "__main__":
    main()