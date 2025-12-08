# solver.py
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg # No ConstraintNorm!
from sklearn.linear_model import Ridge
import config
from utils import SpatialSmoother, TemporalSmoother

def run_mcr_solver(D, C_init, anchor_mask, coords, override_params=None):
    """
    Runs MCR-ALS with Explicit Masking and Physical Constraints.
    """
    # 1. Constraints
    # Apply parameter overrides for optimization
    spatial_alpha = override_params.get('spatial_alpha', config.SPATIAL_ALPHA) if override_params else config.SPATIAL_ALPHA
    anchor_strength = override_params.get('anchor_strength', config.ANCHOR_STRENGTH) if override_params else config.ANCHOR_STRENGTH
    spatial_neighbors = override_params.get('spatial_neighbors', config.SPATIAL_NEIGHBORS) if override_params else config.SPATIAL_NEIGHBORS

    spatial_const = SpatialSmoother(
        coordinates=coords, 
        C_init=C_init,
        anchor_mask=anchor_mask,
        anchor_strength=anchor_strength,
        n_neighbors=spatial_neighbors, 
        alpha=spatial_alpha
    )
    
    temporal_const = TemporalSmoother(
        window_length=config.TEMPORAL_WINDOW, 
        polyorder=config.TEMPORAL_POLY_ORDER
    )
    
    nonneg_const = ConstraintNonneg()

    # 2. Solver Engine
    ridge = Ridge(alpha=config.RIDGE_ALPHA, fit_intercept=False)

    # 3. MCR Loop
    mcr = McrAR(
        c_regr=ridge,
        st_regr=ridge,
        c_constraints=[spatial_const, nonneg_const], 
        st_constraints=[temporal_const], # No Normalization allowed!
        max_iter=config.MAX_ITERATIONS,
        tol_err_change=config.CONVERGENCE_TOL
    )

    mcr.fit(D, C=C_init, verbose=True)

    return mcr.C_opt_, mcr.ST_opt_