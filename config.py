# config.py

# File Paths
INPUT_FILE = 'subset_synthetic_MLCW_layers_differential.csv'
OUTPUT_FILE = 'results_DIFFDISP_1.csv'

# Column Configurations
TOTAL_COL = 'Layer_Total'
STATION_COL = 'STATION'
TIME_COL = 'time'
COORD_COLS = ['X', 'Y']

TARGET_LAYERS = [
    'Layer_1',
    'Layer_2',
    'Layer_3', 
    'Layer_4',
]

# MCR-ALS Solver Parameters
RIDGE_ALPHA = 1e-6        # Very low regularization for clean data
MAX_ITERATIONS = 50       
CONVERGENCE_TOL = 1e-6    

# Smoothing & Anchoring Parameters
SPATIAL_NEIGHBORS = 5     
SPATIAL_ALPHA = 0.3       

# Anchor Strength (0.0 = Free, 1.0 = Rigid)
ANCHOR_STRENGTH = 1 

# Temporal Smoothing (0 = Off, useful for sharp jumps in clean data)
TEMPORAL_WINDOW = 0       
TEMPORAL_POLY_ORDER = 2