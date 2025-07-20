import jax
import types
import warnings
import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

class AdaptiveRejectionSampler:
    def __init__(self, f = None, support = None, log_f = None, abscissae = None):
        self.f = f
        self.support = support
        self.log_f = log_f
        self.h = self.log_f
        self.abscissae = abscissae
        self._jaxified = False
    
    def _jaxify(self):
        if not self._jaxified:
            scope = self.f.__globals__.copy()
            for key in list(scope.keys()):
                val = scope[key]
                if isinstance(val, types.ModuleType):
                    if val.__name__ == 'numpy':
                        scope[key] = jnp
                    elif val.__name__ == 'scipy':
                        scope[key] = jsp
                    elif val.__name__ == 'scipy.stats':
                        scope[key] = jsp.stats
                    elif val.__name__ == 'scipy.special':
                        scope[key] = jsp.special
                    else:
                        pass
                elif hasattr(val, '__module__') and val.__module__.startswith('numpy'):
                    new = getattr(jnp, getattr(val, '__name__', ''), None)
                    if new is not None:
                        scope[key] = new
                elif hasattr(val, '__module__') and val.__module__.startswith('scipy.stats'):
                    new = getattr(jsp.stats, getattr(val, 'name', ''), None)
                    if new is not None:
                        scope[key] = new
                elif hasattr(val, '__module__') and val.__module__.startswith('scipy.special'):
                    new = getattr(jsp.special, getattr(val, 'name', ''), None)
                    if new is not None:
                        scope[key] = new
                elif hasattr(val, '__module__') and val.__module__.startswith('scipy'):
                    new = getattr(jsp, getattr(val, 'name', ''), None)
                    if new is not None:
                        scope[key] = new
                else:
                    pass
            self.f = types.FunctionType(
                self.f.__code__,
                scope,
                name = self.f.__name__,
                argdefs = self.f.__defaults__,
                closure = self.f.__closure__
            )
            self._jaxified = True
    
    def _compute_density(self):
        if self.f is not None:
            pass
        elif self.log_f is not None:
            self.f = lambda x: jnp.exp(self.log_f(x))
        else:
            raise ValueError('Must input density or log density.')
    
    def _compute_log_density(self):
        if self.log_f is not None:
            pass
        elif self.f is not None:
            def h(x):
                return jnp.log(self.f(x))
            self.log_f = h
            self.h = self.log_f
        else:
            raise ValueError('Must input density or log density.')

    def _compute_log_density_derivative(self):
        if self.log_f is not None:
            self.dh = jax.grad(lambda x: jnp.sum(self.h(x)))
        elif self.f is not None:
            self._compute_log_density()
            self.dh = jax.grad(lambda x: jnp.sum(self.h(x)))
        else:
            raise ValueError('Must input density or log density.')

    def _find_mode(self, x0 = 0.0, tol = 20*np.finfo(np.float64).eps):
        try:
            float(x0)
        except TypeError:
            raise TypeError('Initial point must be a numeric value.')
        try:
            self.dh(x0)
            self._jaxified = True
        except jax.errors.TracerArrayConversionError:
            self._jaxify()

        if np.isnan(self.dh(x0)):
            raise ValueError('Initial point not close enough to true mode.')
        
        # Function to provide a more insightful warning than default
        # scipy warning when performing optimization to find mode
        def new_warning(message, category, filename, lineno, file = None, line = None):
            # Check if optimization not making good progress
            if issubclass(category, RuntimeWarning) and 'not making good progress' in message.args[0]:
                print('Warning: Initial point might not be close enough to true mode.')

        # Use warnings.catch_warnings to catch the warning and handle it
        with warnings.catch_warnings():
            # Set the filter to catch RuntimeWarnings
            warnings.simplefilter('always', category = RuntimeWarning)
            warnings.showwarning = new_warning
            mode = sp.optimize.fsolve(self.dh, x0)

        if mode is None:
            # make custom convergence alg later
            pass
        else:
            self.mode = mode
    
    def _initialize_abscissae(self, x0 = None, k = 10, distance = 1.0):
        if self.abscissae is not None:
            return
        
        if not hasattr(self, 'dh'):
            self._compute_log_density_derivative()
        if not callable(self.dh):
            raise TypeError('Input dh must be a callable function.')
        
        try:
            iter(self.support)
        except TypeError:
            raise TypeError('Input support must be an iterable object of length 2.')
        if len(self.support) != 2:
            raise ValueError('Domain must have only a left and right point.')
        if k <= 0 or k % 1 != 0:
            raise ValueError('Input k must be a positive integer.')

        a, b = self.support[0], self.support[1]

        if not ((x0 is None) or (x0 >= a and x0 <= b)):
            raise ValueError('Starting point must be inside domain.')

        if (x0 is None) or (a > -np.inf and b < np.inf):
            if not (a > -np.inf and b < np.inf):
                raise ValueError('Domain must be bounded if no starting point is provided.')
            
            diff = abs(a-b)
            L, R = a + diff/10, b - diff/10
        else:
            if not hasattr(self, 'mode'):
                self._find_mode(x0)

            eps = np.finfo(np.float64).eps
            if a > -np.inf:
                diff = np.abs(self.mode - a)
                R = self.mode + 4*max(diff, distance/2)

                # Only use max in case of computer rounding error resulting
                # in mode - diff/2 being less than a (which can only happen if diff is near 0)
                L = max(self.mode - diff/2, a + eps, a*(1 + eps))
            elif b < np.inf:
                diff = np.abs(self.mode - b)
                L = self.mode - 4*max(diff, distance/2)

                # Only use min in case of computer rounding error resulting
                # in mode + diff/2 being more than b (which can only happen if diff is near 0)
                R = min(self.mode + diff/2, b - eps, b*(1 - eps))
            else:
                # could improve case of fully infinite domain
                L, R = self.mode - distance, self.mode + distance
        
        x_vals = np.linspace(start = L, stop = R, num = k).flatten()

        # We want to avoid generating multiple points such that dh<eps because
        # this could cause dh(xj)-dh(xj+1)=0 and an error could happen later
        dh_vals = self.dh(x_vals)
        where_small = np.where(np.abs(dh_vals) <= 10*eps)[0]

        # Otherwise define 2 ranges and take k total
        # linearly spaced points in those ranges
        L1, R2 = L, R
        while len(where_small) > 0:
            # if this happens it is because f is nearly constant
            # so dh is nearly always 0
            if len(where_small) == len(x_vals):
                raise ValueError('dh must be strictly decreasing')
            # Find the first value that is too small
            j1 = where_small[0]
            # Find the last value that is too small
            j2 = where_small[-1]
            # Set right and left to be the adjacent
            # points that aren't too small
            R1 = x_vals[j1 - 1]
            L2 = x_vals[j2 + 1]
            # If the length of the ranges is 0
            # instead take the midpoint
            if L1 == R1:
                R1 = (L1 + x_vals[j1])/2
            if L2 == R2:
                L2 = (R2 + x_vals[j2])/2
            # Redefine x_vals
            x_vals = np.append(
                np.linspace(
                    start = L1,
                    stop = R1,
                    num = k//2
                ).flatten(),
                np.linspace(
                    start = L2,
                    stop = R2,
                    num = k - k//2
                ).flatten()
            )
            # Check condition on being too small again
            dh_vals = self.dh(x_vals)
            where_small = np.where(np.abs(dh_vals) <= eps)[0]
        
        self.abscissae = x_vals
        self.abscissae_changed = True
    
    def _get_intercepts_and_heights(self, x_vals, h_vals, dh_vals):
        try:
            iter(x_vals)
            iter(h_vals)
            iter(dh_vals)
        except TypeError:
            raise TypeError('All inputs should be iterable objects.')
        if not ((len(x_vals) == len(h_vals)) and (len(h_vals) == len(dh_vals))):
            raise ValueError('x_vals, h_vals, and dh_vals should all be of same size.')
        if not np.all((x_vals >= self.support[0]) & (x_vals <= self.support[1])):
            raise ValueError('All x values should lie inside domain.')
        
        # These are the heights of u_k at x = 0
        # and are used when calculation the
        # intersection points (z) as well
        # as later when sampling from s_k
        heights = h_vals - x_vals*dh_vals

        # These are the intersection points
        z = (heights[1:] - heights[:-1])/(dh_vals[:-1] - dh_vals[1:])
        # Add domain limits to z
        z = np.concat(
            [
                np.array([self.support[0]]),
                z,
                np.array([self.support[1]])
            ]
        )
        return heights, z
    
    def _get_interval_probs(self, z, heights, dh_vals):
        try:
            iter(z)
            iter(heights)
            iter(dh_vals)
        except TypeError:
            raise TypeError('All inputs should be iterable objects.')
        
        # The integrals start with exp(h_j - x_j*dh_j) = exp(heights)
        # where heights represents the intercept with the y axis
        # of the hull function h_j + (x_j - x)*dh_j
        integrals = np.exp(heights)
        # Find where the derivatives are 0
        is_zero = (dh_vals == 0.0)
        # Adjust the integrals as needed
        integrals = np.where(
            is_zero,
            integrals*(z[1:] - z[:-1]),
            integrals*(np.exp(z[1:]*dh_vals) - np.exp(z[:-1]*dh_vals))/dh_vals
        )
        # If less than 0 this is an issue with
        # computer precision so we set it to 0
        integrals = np.maximum(integrals, 0.0)
        probs = integrals/np.sum(integrals)
        return probs

    def _upper(self, z, x_vals, h_vals, dh_vals, x):
        try:
            iter(z)
            iter(x_vals)
            iter(h_vals)
            iter(dh_vals)
            iter(x)
        except TypeError:
            raise TypeError('All inputs should be iterable objects.')
        if not ((len(x_vals) == len(h_vals)) and (len(h_vals) == len(dh_vals))):
            raise ValueError('x_vals, h_vals, and dh_vals should all be of same size.')
        if len(z) != len(x_vals) + 1:
            raise ValueError('z should have one more entry than other arguments')
        if not np.all((x_vals >= z[0]) & (x_vals <= z[-1])) & np.all((x >= z[0]) & (x <= z[-1])):
            raise ValueError('All x values should lie inside domain.')
        j = np.searchsorted(z, x) - 1
        j = np.clip(j, 0, len(z) - 2)

        # Find the upper function at this point
        # NOTE: THIS IS THE UNEXPONENTIATED HULL
        return h_vals[j] + (x - x_vals[j])*dh_vals[j]
    
    def _lower(self, x_vals, h_vals, x):
        try:
            iter(x_vals)
            iter(h_vals)
            iter(x)
        except TypeError:
            raise TypeError('All inputs should be iterable objects.')
        if len(x_vals) != len(h_vals):
            raise ValueError('x_vals and h_vals should be of same size.')
        
        j = np.searchsorted(x_vals, x) - 1
        j = np.clip(j, 0, len(x_vals) - 2)

        # Find which x's lie outside of the x_values
        in_bounds = ((x > x_vals[0]) & (x < x_vals[-1]))
        # Initialize function values
        l = np.full(len(x), -np.inf)
        j_in_bounds = j[in_bounds]

        # Find the lower function at this point
        # NOTE: THIS IS THE UNEXPONENTIATED HULL
        numer = \
            (x_vals[j_in_bounds + 1] - x[in_bounds])*h_vals[j_in_bounds] - \
            (x_vals[j_in_bounds] - x[in_bounds])*h_vals[j_in_bounds + 1]
        denom = x_vals[j_in_bounds + 1] - x_vals[j_in_bounds]
        l[in_bounds] = numer/denom
        return l
    
    def _sample_s_k(self, probs, z, dh_vals, num = 5):
        try:
            iter(probs)
            iter(z)
            iter(dh_vals)
        except TypeError:
            raise TypeError('Point inputs should be iterable objects.')
        if num <= 0 or num % 1 != 0:
            raise ValueError('Input num must be a positive integer.')
        # Choose which range our sample is in
        j = np.random.choice(
            a = len(probs),
            size = num,
            p = probs
        )
        # Find which ranges have dh = 0
        z_lo = z[j]
        z_hi = z[j + 1]
        is_zero = (dh_vals[j] == 0.0)
        is_finite = np.isfinite(z_lo) & np.isfinite(z_hi)
        # We are generating more variables than we need to
        # but this is faster than making a for loop because
        # of numpy vectorized computation

        # Generate uniform random variables for if dh = 0
        x_unif = np.full(num, np.nan)
        x_unif[is_finite] = \
            np.random.uniform(
                low = z_lo[is_finite],
                high = z_hi[is_finite],
                size = np.sum(is_finite)
            )
        
        # Generate standard uniform and use inverse cdf
        # if dh is not zero
        u = np.random.uniform(
            low = 0,
            high = 1,
            size = num
        )
        x_inv_cdf = np.log(
            u*np.exp(z_hi*dh_vals[j]) + \
            (1 - u)*np.exp(z_lo*dh_vals[j]) \
            )/np.where(is_zero, 1.0, dh_vals[j])
        # Create sample according to where dh is zero
        return(np.where(is_zero, x_unif, x_inv_cdf))
    
    def ars(self, n, x0 = None, k = 10, MAXIT = 10e+5):
        eps = np.finfo(np.float64).eps
        tol = 20*eps

        self.sampled = np.array([])
        if n == 0:
            return self.sampled
        
        if not hasattr(self, 'h'):
            self._compute_log_density()
        if not hasattr(self, 'dh'):
            self._compute_log_density_derivative()
        if self.abscissae is None:
            self._initialize_abscissae(x0 = x0, k = k)
        
        iterations = 0
        # Number of points to sample from s_k at once
        num = max(n//10, 1)
        # Make loop to find number of sampled values
        while len(self.sampled) < n:
            iterations += 1
            # Check if maximum iterations reached
            if iterations == MAXIT:
                print('Maximum iterations exceeded, current sampled points returned.')
                return self.sampled
            
            # If sampling a full range could result in more
            # samples than we want decrease the number of
            # samples per iteration
            if n - len(self.sampled) < num:
                num = n - len(self.sampled)
            
            # If the abscissae has changed
            # find the new h and dh values
            if self.abscissae_changed:
                h_vals = self.h(self.abscissae)
                dh_vals = self.dh(self.abscissae)
                # Only keep values for the abscissae that won't
                # cause errors when computing other values

                # Avoid errors when doing np.exp(heights)
                keep_vals = \
                    (np.abs(h_vals) < 200) & \
                    ~(np.isnan(dh_vals)) & \
                    (np.abs(dh_vals) < 200) & \
                    (np.abs(dh_vals) > tol) # Avoid causing there to be 0 values in certain ranges
                
                # If not any we need to raise an error since we have
                # no valid abscissae points in this case we say it is
                # not strictly log concave because the only case this
                # happens is if f is very nearly uniform
                if not np.any(keep_vals):
                    raise ValueError('Input f must be strictly log concave.')
                # If any removed redefine the values
                if not np.all(keep_vals):
                    self.abscissae = self.abscissae[keep_vals]
                    h_vals = h_vals[keep_vals]
                    dh_vals = dh_vals[keep_vals]

                # The way our abscissae generation works the rightmost
                # point is past the mode of f so the derivative of h is
                # negative since h is concave and similarly the leftmost
                # point is before the mode of f so the derivative of h is
                # positive so this is a check for log concavity of f
                if self.support[0] == -np.inf and dh_vals[0] <= 0:
                    raise ValueError('Input f must be strictly log concave.')
                if self.support[1] == np.inf and dh_vals[-1] >= 0:
                    raise ValueError('Input f must be strictly log concave.')

                # Since h is concave dh must be strictly decreasing
                # So this is a check for log concavity of f

                # slow O(n^2) operation
                # if np.min(np.abs(dh_vals[:, None] - dh_vals[None, :])) < tol:
                #     raise ValueError("Input f must be a strictly log concave function.")
                
                # Find the intercept points for the upper hull
                # and the heights of the lines at x = 0
                heights, z = self._get_intercepts_and_heights(self.abscissae, h_vals, dh_vals)

                # Calculate the probability of being in each region
                # according to the density s_k = exp(u_k)/integral(exp(u_k))
                probs = self._get_interval_probs(z, heights, dh_vals)
                self.abscissae_changed = False
            
            # Sample from s_k
            try:
                x_star = self._sample_s_k(probs, z, dh_vals, num)
            except NameError:
                # NameError if already sampled
                h_vals = self.h(self.abscissae)
                dh_vals = self.dh(self.abscissae)
                heights, z = self._get_intercepts_and_heights(self.abscissae, h_vals, dh_vals)
                probs = self._get_interval_probs(z, heights, dh_vals)
                x_star = self._sample_s_k(probs, z, dh_vals, num)

            # Get uniform random value for rejection step
            w = np.random.uniform(
                low = 0.0,
                high = 1.0,
                size = num
            )
            
            # Find upper hull at candidate point
            u = self._upper(
                z, self.abscissae,
                h_vals, dh_vals,
                x_star
            )
            
            # Find lower hull at candidate point
            l = self._lower(
                self.abscissae,
                h_vals,
                x_star
            )
            
            # Check for log concavity of f
            if not np.all((l <= self.h(x_star) + tol) & (self.h(x_star) <= u + tol)):
                raise ValueError('Input f must be a strictly log concave function.')
            
            # Implement rejection steps
            step1 = (w <= np.exp(l - u))
            self.sampled = np.append(self.sampled, x_star[step1])
            step2 = (~step1 & (w <= np.exp(self.h(x_star) - u)))
            self.sampled = np.append(self.sampled, x_star[step2])
            if np.any(~step1):
                self.abscissae_changed = True
            
            # Add x_star if we didn't accept initially
            if self.abscissae_changed:
                self.abscissae = np.unique(np.append(self.abscissae, x_star[~step1]))
        return self.sampled