# utils.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter

class SpatialSmoother:
    """
    Custom Constraint: Enforces Spatial Coherence AND Hard Anchors.
    """
    def __init__(self, coordinates, C_init, n_neighbors=5, alpha=0.3):
        self.alpha = alpha
        self.C_anchor = C_init.copy()
        
        # Identify Anchors (Non-zero initial guesses)
        self.is_anchor = (np.abs(self.C_anchor) > 1e-9)

        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(coordinates)
        self.indices = self.knn.kneighbors(coordinates, return_distance=False)

    def transform(self, C):
        C_next = C.copy()
        
        # 1. Apply Spatial Smoothing
        for i in range(C.shape[0]):
            neighbors = self.indices[i]
            neighbor_avg = np.mean(C[neighbors], axis=0)
            C_next[i] = (1 - self.alpha) * C[i] + self.alpha * neighbor_avg

        # 2. Apply Hard Anchor Reset
        # Force known stations back to their initial ratio values.
        mask = self.is_anchor
        C_next[mask] = self.C_anchor[mask]
        
        return C_next

class TemporalSmoother:
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def transform(self, ST):
        if ST.shape[1] < self.window_length:
            return ST
        return savgol_filter(ST, self.window_length, self.polyorder, axis=1)