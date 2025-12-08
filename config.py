# config.py

# File Paths
INPUT_FILE = 'CRFP_diffdisp.txt'
OUTPUT_FILE = 'robust_mcr_results.csv'

# Column Configurations
TOTAL_COL = 'DIFFDISP'   # The observed total signal
STATION_COL = 'STATION'
TIME_COL = 'time'
COORD_COLS = ['X_TWD97', 'Y_TWD97']

# --- NEW: Explicitly Define Target Variables ---
# Only these specific columns will be solved for.
TARGET_LAYERS = [
    'Layer_1',
    'Layer_2',
    'Layer_3', 
    'Layer_4'
]

# MCR-ALS Solver Parameters
RIDGE_ALPHA = 0.5         
MAX_ITERATIONS = 50       
CONVERGENCE_TOL = 1e-6    

# Smoothing Parameters
SPATIAL_NEIGHBORS = 5     
SPATIAL_ALPHA = 0.3       
TEMPORAL_WINDOW = 5       
TEMPORAL_POLY_ORDER = 2