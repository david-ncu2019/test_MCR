# main.py
import config
import data_loader
import solver
import analytics

def main():
    print(f"--- Starting Upgraded MCR Workflow ---")
    
    # 1. Load
    print("[1/4] Loading Data...")
    df, D_df, D, coords, layer_cols = data_loader.load_and_preprocess(config.INPUT_FILE)
    
    # 2. Init with Mask
    print("[2/4] Generating Sparse Initialization...")
    C_init, anchor_mask = data_loader.generate_ratio_initialization(df, layer_cols, D_df.index)
    
    # 3. Solve with Mask
    print("[3/4] Running MCR-ALS Solver...")
    C_final, ST_final = solver.run_mcr_solver(D, C_init, anchor_mask, coords)
    
    # 4. Save
    print("[4/4] Generating Outputs...")
    analytics.save_predictions(
        df, D_df, C_final, ST_final, layer_cols, config.OUTPUT_FILE
    )
    
    print("\nWorkflow Completed Successfully.")

if __name__ == "__main__":
    main()