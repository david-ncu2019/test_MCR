# solver.py
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNorm, ConstraintNonneg
from sklearn.linear_model import Ridge
import config
from utils import SpatialSmoother, TemporalSmoother

def run_mcr_solver(D, C_init, coords):
    """
    Configures and runs the MCR-ALS optimization.
    """
    # 1. Setup Constraints
    spatial_const = SpatialSmoother(
        coordinates=coords, 
        C_init=C_init,          # <--- FIXED: Now passing the Anchor Matrix
        anchor_strength=config.ANCHOR_STRENGTH,  # <--- PASS CONTROL KNOB HERE
        n_neighbors=config.SPATIAL_NEIGHBORS, 
        alpha=config.SPATIAL_ALPHA
    )
    
    temporal_const = TemporalSmoother(
        window_length=config.TEMPORAL_WINDOW, 
        polyorder=config.TEMPORAL_POLY_ORDER
    )
    
    # norm_const = ConstraintNorm(axis=1)
    nonneg_const = ConstraintNonneg() # <--- FIXED: Enforce positive physics

    # 2. Setup Regressor
    ridge = Ridge(alpha=config.RIDGE_ALPHA, fit_intercept=False)

    # 3. Setup MCR Controller
    mcr = McrAR(
        c_regr=ridge,
        st_regr=ridge,
        # Order: 1. Anchor/Smooth, 2. Enforce Positivity
        c_constraints=[spatial_const, nonneg_const], 
        # st_constraints=[temporal_const, norm_const],
        # CRITICAL FIX: Removed norm_const so S can be large/negative
        st_constraints=[temporal_const],
        max_iter=config.MAX_ITERATIONS,
        tol_err_change=config.CONVERGENCE_TOL
    )

    # 4. Run Optimization
    mcr.fit(D, C=C_init, verbose=True)

    return mcr.C_opt_, mcr.ST_opt_