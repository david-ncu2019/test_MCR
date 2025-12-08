# data_loader.py
import pandas as pd
import numpy as np
import config

def load_and_preprocess(file_path):
    """
    Loads raw data with robust separator detection.
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        if len(df.columns) <= 1:
            raise ValueError("Tab separator failed")
    except:
        print(f"Note: Reading {file_path} as comma-separated...")
        df = pd.read_csv(file_path)

    layer_cols = config.TARGET_LAYERS
    missing = [c for c in layer_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing layers: {missing}")

    # Create Data Matrix D
    D_df = df.pivot_table(index=config.STATION_COL, columns=config.TIME_COL, values=config.TOTAL_COL)
    D = D_df.values
    D = np.nan_to_num(D, nan=0.0)
    
    # Extract Coordinates
    coord_df = df.groupby(config.STATION_COL)[config.COORD_COLS].first()
    coords = coord_df.loc[D_df.index].values

    return df, D_df, D, coords, layer_cols

def generate_ratio_initialization(df, layer_cols, stations):
    """
    Generates C_init AND an Explicit Mask of where data exists.
    """
    n_stations = len(stations)
    n_layers = len(layer_cols)
    
    C_init = np.zeros((n_stations, n_layers))
    anchor_mask = np.zeros((n_stations, n_layers), dtype=bool) # <--- NEW

    station_map = {name: i for i, name in enumerate(stations)}

    for i, layer in enumerate(layer_cols):
        valid_data = df.dropna(subset=[layer, config.TOTAL_COL])
        grouped = valid_data.groupby(config.STATION_COL)
        
        for station_name, group in grouped:
            if station_name in station_map:
                idx = station_map[station_name]
                
                # Calculate Ratio
                ratios = group[layer] / (group[config.TOTAL_COL] + 1e-9)
                local_ratio = np.median(ratios)
                
                C_init[idx, i] = local_ratio
                anchor_mask[idx, i] = True  # <--- Mark as "Known", even if 0.0
                
    return C_init, anchor_mask