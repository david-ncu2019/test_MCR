# config.py

# Data Configuration
INPUT_FILE = "your_data.csv"  # Set to your CSV file path
TARGET_LAYERS = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]
MAP_RESOLUTION = 50

# Expected CSV columns: STATION, X, Y, time, Layer_Total, Layer_1, Layer_2, Layer_3, Layer_4

# Data preprocessing options
HANDLE_MISSING_VALUES = True
MISSING_VALUE_STRATEGY = 'interpolate'  # 'drop', 'interpolate', 'fill_zero'

# Algorithm Parameters
CONVERGENCE_TOLERANCE = 1e-6
MAX_ITERATIONS = 100

# Cross-Validation Parameters
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Bootstrap/Uncertainty Analysis Parameters
BOOTSTRAP_ITERATIONS = 30
SUBSAMPLE_RATIO = 0.8
BOOTSTRAP_PRINT_INTERVAL = 5

# Interpolation Parameters
GRID_PADDING = 1
INTERPOLATION_METHOD_PRIMARY = "cubic"
INTERPOLATION_METHOD_FALLBACK = "nearest"
FILL_VALUE = 0

# Ridge Regression Parameters
RIDGE_FIT_INTERCEPT = False

# Optimization Search Space
PARAM_GRID = {
    'blending_alpha': [0.5, 0.8, 1.0], 
    'ridge_alpha': [1e-6, 1e-3]
}

# Visualization Parameters
FIGURE_SIZE = (24, 6)
STATION_SCATTER_CONFIG = {
    'color': 'white',
    'size': 5,
    'alpha': 0.5,
    'label': 'Stations'
}
COLORMAP_PREDICTION = "turbo"
COLORMAP_UNCERTAINTY = "inferno"
COLORMAP_RELATIVE = "RdYlGn_r"
CV_PLOT_VMIN = 0
CV_PLOT_VMAX = 0.5

# Numerical Stability
EPSILON = 1e-9
