# analytics.py
import pandas as pd
import numpy as np
import config

def save_predictions(df, D_df, C_final, ST_final, layer_cols, output_path):
    """
    Reconstructs the full dataset from C and ST matrices and saves to CSV.
    """
    print("Reconstructing full fields...")
    
    # Lookup dictionaries for fast generation
    coeff_df = pd.DataFrame(C_final, index=D_df.index, columns=layer_cols)
    coeff_dict = coeff_df.to_dict('index')
    
    times = D_df.columns
    time_to_idx = {t: i for i, t in enumerate(times)}
    sig_dict = {layer: ST_final[i] for i, layer in enumerate(layer_cols)}
    
    results = []
    
    # We iterate through the original DF to preserve structure/metadata
    # but fill in the Model predictions
    for idx, row in df.iterrows():
        stat = row[config.STATION_COL]
        t = row[config.TIME_COL]
        
        if t not in time_to_idx or stat not in coeff_dict:
            continue
            
        t_idx = time_to_idx[t]
        
        entry = {
            'STATION': stat,
            'time': t,
            'Observed_Total': row[config.TOTAL_COL]
        }
        
        # Calculate Model Prediction for Total (Sum of parts)
        # Note: In unconstrained MCR, Sum(Parts) != Total, which is what we want.
        
        for layer in layer_cols:
            coeff = coeff_dict[stat][layer]
            sig_val = sig_dict[layer][t_idx]
            pred_val = coeff * sig_val
            
            entry[f'{layer}_Coeff'] = coeff
            entry[f'{layer}_Pred'] = pred_val
            entry[f'{layer}_TimeSignature'] = sig_val
            
            # If we have original sparse data, calculate residual
            if pd.notna(row.get(layer)):
                entry[f'{layer}_Original'] = row[layer]
                entry[f'{layer}_Residual'] = row[layer] - pred_val
        
        results.append(entry)
        
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return final_df