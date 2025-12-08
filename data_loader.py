# data_loader.py
import pandas as pd
import numpy as np
import config

def load_and_preprocess(file_path):
    """
    Loads raw data and prepares the D matrix (Observed Data).
    Returns: df, D_df, D, coords, layer_cols
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
    except:
        df = pd.read_csv(file_path)

    # --- CHANGE: Use Explicit List from Config ---
    # This ensures we only process exactly what you defined
    layer_cols = config.TARGET_LAYERS
    
    # Validation: Check if all requested layers exist in the file
    missing_layers = [c for c in layer_cols if c not in df.columns]
    if missing_layers:
        raise ValueError(f"The following target layers are missing from input file: {missing_layers}")

    # Create Data Matrix D (Stations x Time)
    D_df = df.pivot_table(index=config.STATION_COL, columns=config.TIME_COL, values=config.TOTAL_COL)
    D = D_df.values
    D = np.nan_to_num(D, nan=0.0)
    
    # Extract Coordinates aligned with D rows
    coord_df = df.groupby(config.STATION_COL)[config.COORD_COLS].first()
    coords = coord_df.loc[D_df.index].values

    return df, D_df, D, coords, layer_cols

def generate_ratio_initialization(df, layer_cols, stations):
    """
    Generates the Initial Guess for Coefficients (C_init).
    """
    n_stations = len(stations)
    n_layers = len(layer_cols)
    C_init = np.zeros((n_stations, n_layers))

    for i, layer in enumerate(layer_cols):
        temp_df = df.dropna(subset=[layer, config.TOTAL_COL])
        if len(temp_df) > 0:
            ratios = temp_df[layer] / (temp_df[config.TOTAL_COL] + 1e-9)
            global_ratio = np.median(ratios) 
        else:
            global_ratio = 0.0
        
        C_init[:, i] = global_ratio
        
    return C_init