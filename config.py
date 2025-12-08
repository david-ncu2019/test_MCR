# config.py

# ==========================================
# 1. File Configuration
# ==========================================
# Point to the new realistic (noisy) dataset
INPUT_FILE = 'synthetic_data_realistic.csv'
OUTPUT_FILE = 'realistic_mcr_results.csv'

# Column Names (Must match your CSV headers)
TOTAL_COL = 'Layer_Total'
STATION_COL = 'STATION'
TIME_COL = 'time'
COORD_COLS = ['X', 'Y']

# ==========================================
# 2. Target Definition
# ==========================================
# The realistic generator created 4 physical layers + 1 noise layer.
# We include 'Layer_5' to let the solver catch the random noise 
# separately from the physical signals.
TARGET_LAYERS = [
    'Layer_1',
    'Layer_2',
    'Layer_3', 
    'Layer_4',
    'Layer_5' 
]

# ==========================================
# 3. Solver Parameters (The Physics Tuning)
# ==========================================

# Ridge Regularization (Noise Damping)
# PREVIOUS: 1e-12 (For perfect data)
# NEW: 0.1
# WHY: The data now has noise. We need to penalize wild coefficients 
# to prevent the solver from fitting the random jitter.
RIDGE_ALPHA = 0.1        

# Iteration Limits
MAX_ITERATIONS = 50       
CONVERGENCE_TOL = 1e-6    

# ==========================================
# 4. Constraints (The Knowledge Injection)
# ==========================================

# Spatial Smoothing (Continuity)
SPATIAL_NEIGHBORS = 5     
SPATIAL_ALPHA = 0.3       

# Anchor Strength (Trust in Data)
# PREVIOUS: 1.0 (Rigid/Perfect Trust)
# NEW: 0.6
# WHY: Your initialization (Ratio = Layer/Total) will now be imperfect 
# because of the noise. We trust it 60%, but give the solver 40% freedom 
# to adjust the values to find the true physical pattern.
ANCHOR_STRENGTH = 0.6     

# Temporal Smoothing (Inertia)
# PREVIOUS: 0 (Off, for sharp jumps)
# NEW: 7 (On, Low-Pass Filter)
# WHY: The physical signals (Sine waves, Decay curves) are smooth. 
# The noise is jagged. A window of 7 removes the jagged noise 
# while preserving the smooth geological trends.
TEMPORAL_WINDOW = 7       
TEMPORAL_POLY_ORDER = 2