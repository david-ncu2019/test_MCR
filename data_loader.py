# data_loader.py
import pandas as pd
import numpy as np
import config

def load_and_preprocess(file_path):
    """
    Loads raw data with robust separator detection.
    """
    # 1. Robust Loading
    try:
        df = pd.read_csv(file_path, sep='\t')
        if len(df.columns) <= 1:
            raise ValueError("Tab separator failed")
    except:
        print(f"Note: Reading {file_path} as comma-separated...")
        df = pd.read_csv(file_path)

    # 2. Validation
    layer_cols = config.TARGET_LAYERS
    missing_layers = [c for c in layer_cols if c not in df.columns]
    if missing_layers:
        raise ValueError(f"Missing target layers in file: {missing_layers}")

    # 3. Create Matrices
    D_df = df.pivot_table(index=config.STATION_COL, columns=config.TIME_COL, values=config.TOTAL_COL)
    D = D_df.values
    D = np.nan_to_num(D, nan=0.0)
    
    # Extract Coordinates
    coord_df = df.groupby(config.STATION_COL)[config.COORD_COLS].first()
    coords = coord_df.loc[D_df.index].values

    return df, D_df, D, coords, layer_cols

def generate_ratio_initialization(df, layer_cols, stations):
    """
    Generates a SPARSE Initial Guess (C_init).
    Instead of a global average, we calculate the SPECIFIC ratio for each station.
    """
    n_stations = len(stations)
    n_layers = len(layer_cols)
    
    # Start with Zeros (Unknowns)
    C_init = np.zeros((n_stations, n_layers))

    # Create a mapping from Station Name -> Index for speed
    station_map = {name: i for i, name in enumerate(stations)}

    # Iterate through each layer to find where we have data
    for i, layer in enumerate(layer_cols):
        # Filter for rows where we have BOTH Layer and Total data
        valid_data = df.dropna(subset=[layer, config.TOTAL_COL])
        
        # Group by Station to calculate that station's specific ratio
        # (Some stations might have 1 point, others 50 points)
        grouped = valid_data.groupby(config.STATION_COL)
        
        for station_name, group in grouped:
            if station_name in station_map:
                idx = station_map[station_name]
                
                # Calculate local ratio: Median(Layer / Total)
                # This ensures Station A gets 0.25 and Station B gets 0.40
                ratios = group[layer] / (group[config.TOTAL_COL] + 1e-9)
                local_ratio = np.median(ratios)
                
                # Assign to C_init
                C_init[idx, i] = local_ratio
                
    return C_init