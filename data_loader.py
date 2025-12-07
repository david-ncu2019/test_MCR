# data_loader.py
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import config

def preprocess_data(df):
    """Preprocess DataFrame with flexible column mapping and station filtering."""
    
    # Handle index-based STATION column
    if config.STATION_INDEX_AS_COLUMN and df.index.name == 'STATION':
        df = df.reset_index()
        print("Reset STATION from index to column")
    
    # Rename columns according to mapping
    df = df.rename(columns={
        config.COLUMN_MAPPING['x_col']: 'X',
        config.COLUMN_MAPPING['y_col']: 'Y',
        config.COLUMN_MAPPING['time_col']: 'time',
        config.COLUMN_MAPPING['total_col']: 'Layer_Total'
    })
    print(f"Renamed columns: {config.COLUMN_MAPPING['x_col']}→X, {config.COLUMN_MAPPING['y_col']}→Y, {config.COLUMN_MAPPING['total_col']}→Layer_Total")
    
    # Handle missing values BEFORE filtering stations
    if config.HANDLE_MISSING_VALUES:
        missing_before = df[config.TARGET_LAYERS + ['Layer_Total']].isnull().sum().sum()
        
        if config.MISSING_VALUE_STRATEGY == 'interpolate':
            for station in df['STATION'].unique():
                mask = df['STATION'] == station
                station_data = df.loc[mask].copy().sort_values('time')
                
                for layer in config.TARGET_LAYERS + ['Layer_Total']:
                    if station_data[layer].isnull().any():
                        # Use interpolation with multiple fallback methods
                        station_data[layer] = station_data[layer].interpolate(method='linear')
                        station_data[layer] = station_data[layer].fillna(method='ffill')
                        station_data[layer] = station_data[layer].fillna(method='bfill')
                        # Final fallback to zero if still NaN
                        station_data[layer] = station_data[layer].fillna(0)
                
                df.loc[mask] = station_data
        elif config.MISSING_VALUE_STRATEGY == 'drop':
            df = df.dropna(subset=config.TARGET_LAYERS + ['Layer_Total'])
        elif config.MISSING_VALUE_STRATEGY == 'fill_zero':
            df[config.TARGET_LAYERS + ['Layer_Total']] = df[config.TARGET_LAYERS + ['Layer_Total']].fillna(0)
        
        # Ensure no NaN values remain
        df[config.TARGET_LAYERS + ['Layer_Total']] = df[config.TARGET_LAYERS + ['Layer_Total']].fillna(0)
        
        missing_after = df[config.TARGET_LAYERS + ['Layer_Total']].isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
    
    # Filter stations with insufficient observations AFTER handling missing values
    station_counts = df['STATION'].value_counts()
    valid_stations = station_counts[station_counts >= config.MIN_OBSERVATIONS_PER_STATION]
    excluded_stations = station_counts[station_counts < config.MIN_OBSERVATIONS_PER_STATION]
    
    if len(excluded_stations) > 0:
        print(f"Excluding {len(excluded_stations)} stations with <{config.MIN_OBSERVATIONS_PER_STATION} observations:")
        for station, count in excluded_stations.items():
            print(f"  {station}: {count} observations")
        df = df[df['STATION'].isin(valid_stations.index)]
    
    print(f"Kept {len(valid_stations)} stations with ≥{config.MIN_OBSERVATIONS_PER_STATION} observations")
    
    return df.sort_values(['STATION', 'time']).reset_index(drop=True)


def load_data(filepath):
    """Load and preprocess data with flexible column mapping."""
    df = pd.read_csv(filepath)
    df = preprocess_data(df)
    
    time_steps = np.sort(df["time"].unique())
    station_names = df["STATION"].unique()

    # Create anchor coordinates dictionary
    anchor_coords = {}
    for station in station_names:
        row = df[df["STATION"] == station].iloc[0]
        anchor_coords[station] = (row["X"], row["Y"])

    # Create pivot tables with explicit NaN handling
    total_signal_matrix = df.pivot(index="time", columns="STATION", values="Layer_Total")
    total_signal_matrix = total_signal_matrix.fillna(0)  # Ensure no NaNs
    
    anchor_signals_dict = {}
    for layer in config.TARGET_LAYERS:
        pivot_table = df.pivot(index="time", columns="STATION", values=layer)
        pivot_table = pivot_table.fillna(0)  # Ensure no NaNs
        anchor_signals_dict[layer] = pivot_table

    return time_steps, station_names, anchor_coords, total_signal_matrix, anchor_signals_dict


def generate_map_grid(anchor_coords, resolution):
    x_vals = [c[0] for c in anchor_coords.values()]
    y_vals = [c[1] for c in anchor_coords.values()]
    xi = np.linspace(min(x_vals) - config.GRID_PADDING, max(x_vals) + config.GRID_PADDING, resolution)
    yi = np.linspace(min(y_vals) - config.GRID_PADDING, max(y_vals) + config.GRID_PADDING, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    return np.column_stack([Xi.ravel(), Yi.ravel()]), Xi, Yi


def build_total_map(total_matrix, anchor_coords, station_names, all_pixels):
    """Build total map with robust NaN handling."""
    num_times = total_matrix.shape[0]
    total_map = np.zeros((num_times, len(all_pixels)))
    station_locs = np.array([anchor_coords[s] for s in station_names])

    for t_idx in range(num_times):
        vals = total_matrix.iloc[t_idx].values
        
        # Handle NaN values more robustly
        valid_mask = ~np.isnan(vals) & np.isfinite(vals)
        
        if valid_mask.sum() < 3:
            # If too few valid points, use nearest neighbor interpolation
            if valid_mask.sum() > 0:
                interp_vals = griddata(
                    station_locs[valid_mask], vals[valid_mask], all_pixels, 
                    method='nearest', fill_value=0
                )
            else:
                # If no valid points at all, fill with zeros
                interp_vals = np.zeros(len(all_pixels))
        else:
            # Try primary interpolation method
            try:
                interp_vals = griddata(
                    station_locs[valid_mask], vals[valid_mask], all_pixels, 
                    method=config.INTERPOLATION_METHOD_PRIMARY, fill_value=0
                )
            except:
                # Fallback to linear if cubic fails
                try:
                    interp_vals = griddata(
                        station_locs[valid_mask], vals[valid_mask], all_pixels, 
                        method='linear', fill_value=0
                    )
                except:
                    # Final fallback to nearest neighbor
                    interp_vals = griddata(
                        station_locs[valid_mask], vals[valid_mask], all_pixels, 
                        method='nearest', fill_value=0
                    )
            
            # Handle any remaining NaN values
            mask_nan = np.isnan(interp_vals)
            if np.any(mask_nan):
                try:
                    fallback_vals = griddata(
                        station_locs[valid_mask], vals[valid_mask], all_pixels[mask_nan], 
                        method=config.INTERPOLATION_METHOD_FALLBACK, fill_value=0
                    )
                    interp_vals[mask_nan] = fallback_vals
                except:
                    # If fallback also fails, fill with zeros
                    interp_vals[mask_nan] = 0
        
        # Final safety check - replace any remaining NaN/inf values
        interp_vals = np.where(np.isfinite(interp_vals), interp_vals, 0)
        total_map[t_idx, :] = interp_vals
    
    return total_map
