# config.py

# ==========================================
# 1. File Configuration
# ==========================================
INPUT_FILE = 'synthetic_data_realistic.csv'
OUTPUT_FILE = 'realistic_mcr_results.csv'

# --- NEW: User explicitly defines the parameter file ---
# Change this name when you switch datasets (e.g., 'params_real_data.json')
# to avoid mixing up parameters between different experiments.
OPTIMAL_PARAMS_FILE = 'optimal_params_realistic.json'

# Column Names
TOTAL_COL = 'Layer_Total'
STATION_COL = 'STATION'
TIME_COL = 'time'
COORD_COLS = ['X', 'Y']

# ==========================================
# 2. Target Definition
# ==========================================
TARGET_LAYERS = [
    'Layer_1',
    'Layer_2',
    'Layer_3', 
    'Layer_4',
    'Layer_5' 
]

# ==========================================
# 3. Solver Parameters (Robust Mode)
# ==========================================
RIDGE_ALPHA = 0.1        
MAX_ITERATIONS = 50       
CONVERGENCE_TOL = 1e-6    

# ==========================================
# 4. Constraints (Robust Mode)
# ==========================================
SPATIAL_NEIGHBORS = 5     
SPATIAL_ALPHA = 0.3       
ANCHOR_STRENGTH = 0.6     
TEMPORAL_WINDOW = 7       
TEMPORAL_POLY_ORDER = 2

# ==========================================
# 5. Optimization Search Grids
# ==========================================
# Define the ranges you want to test when running optimization.

PARAM_GRID_QUICK = {
    'spatial_alpha': [0.1, 0.3, 0.5],
    'anchor_strength': [0.5, 0.7, 1.0],
    'spatial_neighbors': [5]
}

PARAM_GRID_FULL = {
    'spatial_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'anchor_strength': [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    'spatial_neighbors': [3, 5, 8, 10]
}