# data_loader.py
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import config

def load_data(filepath):
    """Load data from CSV file with standard column structure."""
    df = pd.read_csv(filepath)
    
    # Handle missing values if configured
    if config.HANDLE_MISSING_VALUES:
        missing_before = df[config.TARGET_LAYERS].isnull().sum().sum()
        
        if config.MISSING_VALUE_STRATEGY == 'interpolate':
            for station in df['STATION'].unique():
                mask = df['STATION'] == station
                station_data = df.loc[mask].copy().sort_values('time')
                
                for layer in config.TARGET_LAYERS:
                    if station_data[layer].isnull().any():
                        station_data[layer] = station_data[layer].interpolate()
                        station_data[layer] = station_data[layer].fillna(method='ffill').fillna(method='bfill')
                
                df.loc[mask] = station_data
        elif config.MISSING_VALUE_STRATEGY == 'drop':
            df = df.dropna(subset=config.TARGET_LAYERS)
        elif config.MISSING_VALUE_STRATEGY == 'fill_zero':
            df[config.TARGET_LAYERS] = df[config.TARGET_LAYERS].fillna(0)
        
        missing_after = df[config.TARGET_LAYERS].isnull().sum().sum()
        print(f"Missing values handled: {missing_before} → {missing_after}")
    
    time_steps = np.sort(df["time"].unique())
    station_names = df["STATION"].unique()

    # Create anchor coordinates dictionary
    anchor_coords = {}
    for station in station_names:
        row = df[df["STATION"] == station].iloc[0]
        anchor_coords[station] = (row["X"], row["Y"])

    # Create pivot tables
    total_signal_matrix = df.pivot(index="time", columns="STATION", values="Layer_Total")
    
    anchor_signals_dict = {}
    for layer in config.TARGET_LAYERS:
        anchor_signals_dict[layer] = df.pivot(index="time", columns="STATION", values=layer)

    return time_steps, station_names, anchor_coords, total_signal_matrix, anchor_signals_dict


def generate_map_grid(anchor_coords, resolution):
    x_vals = [c[0] for c in anchor_coords.values()]
    y_vals = [c[1] for c in anchor_coords.values()]
    xi = np.linspace(min(x_vals) - config.GRID_PADDING, max(x_vals) + config.GRID_PADDING, resolution)
    yi = np.linspace(min(y_vals) - config.GRID_PADDING, max(y_vals) + config.GRID_PADDING, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    return np.column_stack([Xi.ravel(), Yi.ravel()]), Xi, Yi


def build_total_map(total_matrix, anchor_coords, station_names, all_pixels):
    num_times = total_matrix.shape[0]
    total_map = np.zeros((num_times, len(all_pixels)))
    station_locs = np.array([anchor_coords[s] for s in station_names])

    for t_idx in range(num_times):
        vals = total_matrix.iloc[t_idx].values
        valid_mask = ~np.isnan(vals)
        
        if valid_mask.sum() < 3:
            total_map[t_idx, :] = griddata(
                station_locs[valid_mask], vals[valid_mask], all_pixels, 
                method=config.INTERPOLATION_METHOD_FALLBACK, fill_value=config.FILL_VALUE
            )
        else:
            total_map[t_idx, :] = griddata(
                station_locs[valid_mask], vals[valid_mask], all_pixels, 
                method=config.INTERPOLATION_METHOD_PRIMARY, fill_value=config.FILL_VALUE
            )
            mask = np.isnan(total_map[t_idx, :])
            if np.any(mask):
                total_map[t_idx, mask] = griddata(
                    station_locs[valid_mask], vals[valid_mask], all_pixels[mask], 
                    method=config.INTERPOLATION_METHOD_FALLBACK
                )
    return total_map
