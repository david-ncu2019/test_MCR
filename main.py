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
    
    print("\n[?] Optimization Required or Requested.")
    if input("Run parameter optimization now? (y/n): ").lower() == 'y':
        print("Running optimization...")
        best_params, _ = run_quick_optimization()
        return best_params
    return None

def main():
    print(f"--- Starting MCR Workflow ---")
    print(f"Target Input:   {config.INPUT_FILE}")
    print(f"Parameter File: {config.OPTIMAL_PARAMS_FILE}")

    # 1. Use the User-Defined Parameter File
    param_file = config.OPTIMAL_PARAMS_FILE
    
    optimized_params = None

    # 2. Check for EXISTING parameters
    if os.path.exists(param_file):
        print(f"\n[+] Found existing parameters.")
        with open(param_file, 'r') as f:
            optimized_params = json.load(f)
        print(f"Using: {optimized_params}")
        
        # Option to re-run if the user wants
        if input("Recalculate parameters anyway? (y/n): ").lower() == 'y':
            optimized_params = run_optimization()
            if optimized_params:
                with open(param_file, 'w') as f:
                    json.dump(optimized_params, f, indent=2)
                    print(f"[+] Updated parameters saved to {param_file}")
    else:
        # No parameters found, prompt to run
        print(f"\n[-] Parameter file not found: {param_file}")
        optimized_params = run_optimization()
        
        if optimized_params:
            with open(param_file, 'w') as f:
                json.dump(optimized_params, f, indent=2)
            print(f"[+] New parameters saved to {param_file}")
    
    # 3. Load Data
    print("\n[1/4] Loading Data...")
    df, D_df, D, coords, layer_cols, coord_df = data_loader.load_and_preprocess(config.INPUT_FILE)
    print(f"Loaded {len(coord_df)} stations.")
    
    # 4. Initialize
    print("[2/4] Generating Sparse Initialization...")
    C_init, anchor_mask = data_loader.generate_ratio_initialization(df, layer_cols, D_df.index)
    
    # 5. Solve (Using the loaded/optimized params)
    print("[3/4] Running MCR-ALS Solver...")
    C_final, ST_final = solver.run_mcr_solver(D, C_init, anchor_mask, coords, optimized_params)
    
    # 6. Save
    print("[4/4] Generating Outputs...")
    analytics.save_predictions(
        df, D_df, C_final, ST_final, layer_cols, config.OUTPUT_FILE, coord_df
    )
    
    print("\nWorkflow Completed Successfully.")

if __name__ == "__main__":
    main()