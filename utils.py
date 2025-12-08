# utils.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter

class SpatialSmoother:
    """
    Custom Constraint: Enforces Spatial Coherence with ELASTIC Anchors.
    """
    def __init__(self, coordinates, C_init, anchor_mask, anchor_strength=0.5, n_neighbors=5, alpha=0.3):
        self.alpha = alpha
        self.anchor_strength = anchor_strength
        self.C_anchor = C_init.copy()
        self.mask = anchor_mask.copy()

        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(coordinates)
        self.indices = self.knn.kneighbors(coordinates, return_distance=False)

    def transform(self, C):
        C_next = C.copy()
        
        # 1. Physics: Smooth everything based on neighbors
        for i in range(C.shape[0]):
            neighbors = self.indices[i]
            neighbor_avg = np.mean(C[neighbors], axis=0)
            C_next[i] = (1 - self.alpha) * C[i] + self.alpha * neighbor_avg

        # 2. Data: Blend known anchors back towards truth
        C_next[self.mask] = (
            (1 - self.anchor_strength) * C_next[self.mask] + 
            self.anchor_strength * self.C_anchor[self.mask]
        )
        
        return C_next

class TemporalSmoother:
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def transform(self, ST):
        # --- CRITICAL FIX: Skip if window is 0 or too small ---
        if self.window_length <= 2 or ST.shape[1] < self.window_length:
            return ST 
        # ------------------------------------------------------
        
        return savgol_filter(ST, self.window_length, self.polyorder, axis=1)