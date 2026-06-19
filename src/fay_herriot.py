import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FayHerriotModel:
    """
    Implementation of the Fay-Herriot area-level model for Small Area Estimation.
    Used to compute Empirical Best Linear Unbiased Predictors (EBLUP).
    """
    def __init__(self):
        self.beta = None
        self.sigma_v = None
        self.eblup = None
        self.gamma = None
        
    def fit(self, X: pd.DataFrame, y_direct: pd.Series, sampling_variance: pd.Series):
        """
        Fits the Fay-Herriot model using the method of moments or REML for variance components.
        """
        logging.info("Fitting Fay-Herriot model...")
        
        # Add intercept to auxiliary variables
        X_mat = sm.add_constant(X)
        
        # 1. Initial OLS to estimate beta
        ols_model = sm.OLS(y_direct, X_mat).fit()
        
        # 2. Estimate spatial variance (sigma_v^2) - Simplified moment estimator
        mse_ols = ols_model.mse_resid
        mean_sampling_var = np.mean(sampling_variance)
        self.sigma_v = max(0, mse_ols - mean_sampling_var)
        logging.info(f"Estimated spatial variance component (sigma_v^2): {self.sigma_v:.4f}")
        
        # 3. Calculate shrinkage factor (Gamma)
        self.gamma = self.sigma_v / (self.sigma_v + sampling_variance)
        
        # 4. WLS to re-estimate beta with heteroscedastic weights
        weights = 1.0 / (self.sigma_v + sampling_variance)
        wls_model = sm.WLS(y_direct, X_mat, weights=weights).fit()
        self.beta = wls_model.params
        
        # 5. Compute EBLUPs (Empirical Best Linear Unbiased Predictors)
        synthetic_est = X_mat.dot(self.beta)
        self.eblup = self.gamma * y_direct + (1 - self.gamma) * synthetic_est
        
        logging.info("EBLUP computation completed.")
        return self.eblup

    def get_diagnostics(self) -> pd.DataFrame:
        """Returns the diagnostic shrinkage factors."""
        if self.gamma is None:
            raise ValueError("Model has not been fitted yet.")
        return pd.DataFrame({'Shrinkage_Gamma': self.gamma})