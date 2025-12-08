# utils.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter

class SpatialSmoother:
    """
    Custom Constraint: Enforces Spatial Coherence with ELASTIC Anchors.
    """
    def __init__(self, coordinates, C_init, anchor_strength=0.5, n_neighbors=5, alpha=0.3):
        self.alpha = alpha
        self.anchor_strength = anchor_strength  # How strongly to pull back to data
        self.C_anchor = C_init.copy()
        
        # Identify Anchors (Non-zero initial guesses)
        self.is_anchor = (np.abs(self.C_anchor) > 1e-9)

        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(coordinates)
        self.indices = self.knn.kneighbors(coordinates, return_distance=False)

    def transform(self, C):
        C_next = C.copy()
        
        # 1. Apply Spatial Smoothing (The "Physics" Pull)
        # Pulls values towards their neighbors
        for i in range(C.shape[0]):
            neighbors = self.indices[i]
            neighbor_avg = np.mean(C[neighbors], axis=0)
            C_next[i] = (1 - self.alpha) * C[i] + self.alpha * neighbor_avg

        # 2. Apply Soft Anchor (The "Data" Pull)
        # Instead of replacing (C = Anchor), we Blend (C = Mix of Current & Anchor)
        # Formula: New = (1 - Strength) * Model_Guess + (Strength) * Data_Anchor
        
        mask = self.is_anchor
        
        # The Blend:
        # If Strength is 0.1, we keep 90% of the Solver's new idea, add 10% Data.
        # If Strength is 0.9, we force it 90% back to the Data.
        C_next[mask] = (
            (1 - self.anchor_strength) * C_next[mask] + 
            self.anchor_strength * self.C_anchor[mask]
        )
        
        return C_next

class TemporalSmoother:
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def transform(self, ST):
        if ST.shape[1] < self.window_length:
            return ST
        return savgol_filter(ST, self.window_length, self.polyorder, axis=1)