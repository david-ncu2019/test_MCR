# config.py

INPUT_FILE = "subset_synthetic_MLCW_layers_differential.csv"
TARGET_LAYERS = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]
MAP_RESOLUTION = 50
CONVERGENCE_TOLERANCE = 1e-6

# The Search Space for the Optimizer
PARAM_GRID = {
    'blending_alpha': [0.5, 0.8, 1.0], 
    'ridge_alpha': [1e-6, 1e-3]
}