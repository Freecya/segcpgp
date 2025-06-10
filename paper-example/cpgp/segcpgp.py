import tensorflow as tf
from scipy.stats.distributions import chi2
import sys
import os
import gpflow as gpf
import numpy as np
np.random.seed(42)
f64 = gpf.utilities.to_default_float
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

from cpgp.spectral_mixture import SpectralMixture

def get_kernel(name, X, y):
    """Get kernel based on name. X and y need to be passed for the spectral mixture GMM to fit on.

    Arguments:
        name -- name of the kernel to be returned.
        X -- index.
        y -- observations.

    Returns:
        k -- gpflow Kernel object. 
    """
    name = name.lower()
    if "noise" in name and "spectral" not in name:
        k = gpf.kernels.White(0.1)
    if "spectral-" in name:
        q = int(name.split("-")[1])
        k = SpectralMixture(q, x=X, y=y)
    if "linear" in name:
        k = gpf.kernels.Linear()  
    if "matern" in name:
        k = gpf.kernels.Matern52()  
    if "constant" in name:
        k = gpf.kernels.Constant()
    if "rbf" in name:
        k = gpf.kernels.RBF()
    if "per" in name:
        k = gpf.kernels.Periodic(gpf.kernels.RBF(), 1)

    return k


class SegCPGP():
    """Implements Segmenting Changepoint Gaussian Processes"""
    def __init__(self, pval=0.1, attempts=3, logging=False) -> None:
        self.LOCS = []
        self.STEEPNESS = []
        self.TESTED = []
        self.lrts = []
        self.logging = logging
        self.pval = pval
        self.attempts = attempts
        self.lrts = []
        self.verbose = False

    def fit(self, X, y, base_kernel_name="constant", custom_kernel=None, verbose=True):
        """Fit SegCPGP

        Arguments:
            X -- index
            y -- observations

        Keyword Arguments:
            base_kernel_name -- kernel used in the CP kernel (default: {"constant"})

        Returns:
            LOCS, STEEPNESS -- locations and associated steepnesses. 
        """
        self.verbose = verbose
        results = self.call(X, y, base_kernel_name, custom_kernel)
        return results

    def get_high_likelihood_model(self, X, y, model_name, base_kernel_name, custom_kernel):
        """The LML has no guarantees that it does not end up in a local optimum.
        We select the model with the highest likelihood after a number of attempts. 

        Arguments:
            attempts -- number of attempts to make

        Returns: 
            model -- model with the highest likelihood.
        """
        def get_model(X, y, model_name, base_kernel_name, custom_kernel):
            """Utility function for getting either a GPR or a CP based on the model_name string."""
            if not custom_kernel:
                kernels = [get_kernel(base_kernel_name, X, y), get_kernel(
                    base_kernel_name, X, y)]
            else:
                kernels = [custom_kernel, custom_kernel]
            if model_name == "cp":
                model = gpf.models.GPR((X, y), kernel=gpf.kernels.ChangePoints(
                    kernels, locations=[np.random.randint(X.min(), X.max())], steepness=[1]))
            else:
                model = gpf.models.GPR((X, y), kernel=kernels[0])
            return model

        models = []
        for _ in range(self.attempts):
            model = get_model(X, y, model_name, base_kernel_name, custom_kernel)
            optimizer = gpf.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables)
            models.append(model)

        # Find model with highest likelihood.
        models.sort(key=lambda m: m.log_marginal_likelihood())
        model = models[-1]
        return model

    def call(self, X, y, base_kernel_name="constant", custom_kernel=None):
        """Recursive bisecting function. 

        Arguments:
            X -- index
            y -- observations
            base_kernel_name -- kernel used in the CP kernel

        Returns:
            LOCS, STEEPNESSES -- changepoint LOCationS and their associated STEEPNESSES
        """
        if self.verbose:
            print(X.min(), X.max())
        models = {}
        
        # Get single and changepoint model
        for model_name in ["cp", "gpr"]:
            model = self.get_high_likelihood_model(
                X, y, model_name, base_kernel_name, custom_kernel)
            models[model_name] = model

        cp = models["cp"]
        gpr = models["gpr"]
        location = cp.kernel.locations.numpy()
        steep = cp.kernel.steepness.numpy()

        # Compute LRT
        LRT = -2 * (gpr.log_marginal_likelihood() -
                    cp.log_marginal_likelihood())
        if self.logging:
            self.lrts.append(LRT)
        df = len(cp.trainable_parameters) - len(gpr.trainable_parameters)
        p = chi2.sf(LRT, df)
        
        if self.verbose:
            print("p", p, "df", df, "location", location, "steepness", steep)
        
        # Record all hypothesis tests
        test = [float(location), float(steep), p,
                float(X[0]), float(X[-1]), cp, gpr]
        self.TESTED.append(test)

        # Try splitting
        # The null model is favored and we are done.
        if p > self.pval or np.isnan(p):
            return self.LOCS, self.STEEPNESS
        
        else:  # Split the signal
            # Check if t_0 is out of bounds
            if min(X) > location or location > max(X):
                return self.LOCS, self.STEEPNESS

            # Check if location not found, else return.
            if int(location) not in list(map(int, self.LOCS)):
                self.LOCS.append(location)
                self.STEEPNESS.append(steep)
            else:
                return self.LOCS, self.STEEPNESS

            # Margin around changepoint to avoid detecting same changepoint multiple times
            try:
                epsilon = 5
                b2 = 5
                if location - epsilon <= X.min(): # Edge case: if location is at the edge of the signal, do not use margin
                    b1 = 0
                if location + epsilon >= X.max(): # Edge case: ""
                    b2 = 0

                split_left = list(map(int, X)).index(int(location-epsilon))
                split_right = list(map(int, X)).index(int(location+epsilon))

                X_left, X_right = X[:split_left], X[split_right:]
                y_left, y_right = y[:split_left], y[split_right:]
                
            except ValueError:
                split = list(map(int, X)).index(int(location))
                X_left, X_right = X[:split], X[split:]
                y_left, y_right = y[:split], y[split:]
                
            # Recurse if signal is long enough.
            if len(X_left) > 2:
                self.call(X_left, y_left, base_kernel_name)
            if len(X_right) > 2:
                self.call(X_right, y_right, base_kernel_name)
        return self.LOCS, self.STEEPNESS
