# config.py

# File Paths
INPUT_FILE = 'subset_synthetic_MLCW_layers_differential.csv'
OUTPUT_FILE = 'results_DIFFDISP_3.csv'

# Column Configurations
TOTAL_COL = 'Layer_Total'
STATION_COL = 'STATION'
TIME_COL = 'time'
COORD_COLS = ['X', 'Y']

TARGET_LAYERS = [
    'Layer_1',
    'Layer_2',
    'Layer_3', 
    'Layer_4'
]

# MCR-ALS Solver Parameters
RIDGE_ALPHA = 1e-6         
MAX_ITERATIONS = 50       
CONVERGENCE_TOL = 1e-6    

# Smoothing & Anchoring Parameters
SPATIAL_NEIGHBORS = 5     
SPATIAL_ALPHA = 0.3       

# --- NEW: Anchor Strength (0.0 to 1.0) ---
# 0.9 = Strong trust in input data (Stiff)
# 0.1 = Weak trust in input data (Flexible)
ANCHOR_STRENGTH = 0.5     

TEMPORAL_WINDOW = 0       
TEMPORAL_POLY_ORDER = 2