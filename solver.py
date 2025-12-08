# solver.py
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNorm
from sklearn.linear_model import Ridge
import config
from utils import SpatialSmoother, TemporalSmoother

def run_mcr_solver(D, C_init, coords):
    """
    Configures and runs the MCR-ALS optimization.
    """
    # 1. Setup Constraints
    spatial_const = SpatialSmoother(
        coords, 
        n_neighbors=config.SPATIAL_NEIGHBORS, 
        alpha=config.SPATIAL_ALPHA
    )
    
    temporal_const = TemporalSmoother(
        window_length=config.TEMPORAL_WINDOW, 
        polyorder=config.TEMPORAL_POLY_ORDER
    )
    
    norm_const = ConstraintNorm(axis=1) # Normalize signatures

    # 2. Setup Regressor
    ridge = Ridge(alpha=config.RIDGE_ALPHA, fit_intercept=False)

    # 3. Setup MCR Controller
    mcr = McrAR(
        c_regr=ridge,
        st_regr=ridge,
        c_constraints=[spatial_const],
        st_constraints=[temporal_const, norm_const],
        max_iter=config.MAX_ITERATIONS,
        tol_err_change=config.CONVERGENCE_TOL
    )

    # 4. Run Optimization
    # We provide C=C_init, so it solves for ST first, then C...
    mcr.fit(D, C=C_init, verbose=True)

    return mcr.C_opt_, mcr.ST_opt_