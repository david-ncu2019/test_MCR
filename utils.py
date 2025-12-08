# utils.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter

class SpatialSmoother:
    """
    Custom pyMCR Constraint: Enforces spatial coherence.
    Each station's coefficient is blended with its nearest neighbors.
    """
    def __init__(self, coordinates, n_neighbors=5, alpha=0.3):
        self.alpha = alpha
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(coordinates)
        self.indices = self.knn.kneighbors(coordinates, return_distance=False)

    def transform(self, C):
        # C shape: (Stations, Layers)
        C_smooth = C.copy()
        for i in range(C.shape[0]):
            neighbors = self.indices[i]
            neighbor_avg = np.mean(C[neighbors], axis=0)
            C_smooth[i] = (1 - self.alpha) * C[i] + self.alpha * neighbor_avg
        return C_smooth

class TemporalSmoother:
    """
    Custom pyMCR Constraint: Enforces temporal smoothness.
    Uses Savitzky-Golay filter to remove high-frequency noise from Time Signatures.
    """
    def __init__(self, window_length=5, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder

    def transform(self, ST):
        # ST shape: (Layers, Time)
        if ST.shape[1] < self.window_length:
            return ST
        return savgol_filter(ST, self.window_length, self.polyorder, axis=1)