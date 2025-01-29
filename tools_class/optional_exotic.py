import pandas as pd
import numpy as np
from scipy.stats import norm

class Optional_Excotic:
    def __init__(self):
        pass


    def black_scholes_call(self, S, K, T, r, sigma):
        if T == 0:
            return max(0, S - K)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


    def call_spread(self, S, K1, K2, T, r, sigma):
        """
        Price a Call Spread: Buy Call(K1) - Sell Call(K2)
        """
        call1 = self.black_scholes_call(S, K1, T, r, sigma)
        call2 = self.black_scholes_call(S, K2, T, r, sigma)
        return call1 - call2

    def butterfly_spread(self, S, K1, K2, K3, T, r, sigma):
        """
        Price a Butterfly Spread
        """
        call1 = self.black_scholes_call(S, K1, T, r, sigma)
        call2 = 2 * self.black_scholes_call(S, K2, T, r, sigma)
        call3 = self.black_scholes_call(S, K3, T, r, sigma)
        return call1 - call2 + call3


    # Monte Carlo simulation for Black-Scholes trajectories
    def simulate_black_scholes(self, S0, T, r, sigma, n_paths, n_steps, seed=None): # Added seed as an argument with default value None
        if seed is not None: # Check if seed was provided
            np.random.seed(seed) # If yes, set the seed
        else:
            np.random.seed(102)
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = S0
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_paths)
            paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return paths

    # Asian option pricing with specific averaging window
    def asian_option_monte_carlo(self, S0, K, T, r, sigma, n_paths, n_steps, delta, option_type="call", seed=None):
        # Pass the seed to the simulation function
        paths = self.simulate_black_scholes(S0, T, r, sigma, n_paths, n_steps, seed=seed)
        start_index = int((T - delta) * n_steps / T)
        averages = np.mean(paths[start_index:], axis=0)  # Average over [T-delta, T]
        if option_type == "call":
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)
        expectation = np.mean(paths[-1] - K)
        return np.exp(-r * T) * np.mean(payoffs), paths, averages, expectation


    # Barrier option pricing
    def barrier_option(self, S0, K, T, r, sigma, n_paths, n_steps, barrier, option_type="call", barrier_type="up-and-out", seed=None):
        paths = self.simulate_black_scholes(S0, T, r, sigma, n_paths, n_steps, seed=seed)
        if barrier_type == "up-and-out":
            invalid = np.any(paths > barrier, axis=0)
        elif barrier_type == "down-and-out":
            invalid = np.any(paths < barrier, axis=0)
        else:
            raise ValueError("Unsupported barrier type")

        if option_type == "call":
            payoffs = np.maximum(paths[-1] - K, 0)
        else:
            payoffs = np.maximum(K - paths[-1], 0)

        payoffs[invalid] = 0  # Invalidate payoffs for paths breaching the barrier
        return np.exp(-r * T) * np.mean(payoffs)


    def calculate_greeks_asiatique(self, option_price_fn, S0, K, T, r, sigma, delta=0.25, n_paths=1000, n_steps=100, epsilon=1e-4, greek="delta", seed=None):
        if greek == "delta":
            price_up = option_price_fn(S0 + epsilon, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            price = option_price_fn(S0, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            return (price_up - price) / epsilon

        elif greek == "gamma":
            price_up = option_price_fn(S0 + epsilon, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            price = option_price_fn(S0, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            price_down = option_price_fn(S0 - epsilon, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            return (price_up - 2 * price + price_down) / (epsilon ** 2)

        elif greek == "vega":
            price_up = option_price_fn(S0, K, T, r, sigma + epsilon, n_paths, n_steps, delta, seed=seed)[0]
            price = option_price_fn(S0, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            return (price_up - price) / epsilon

        elif greek == "theta":
            price_up = option_price_fn(S0, K, T - epsilon, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            price = option_price_fn(S0, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            return (price - price_up) / epsilon

        elif greek == "rho":
            price_up = option_price_fn(S0, K, T, r + epsilon, sigma, n_paths, n_steps, delta, seed=seed)[0]
            price = option_price_fn(S0, K, T, r, sigma, n_paths, n_steps, delta, seed=seed)[0]
            return (price_up - price) / epsilon

        else:
            raise ValueError("Unsupported Greek type")

        
    def calculate_greeks_barrier(self, option_price_fn, S0, K, T, r, sigma, barrier, option_type="call", barrier_type="up-and-out", delta=0.25, n_paths=1000, n_steps=100, epsilon=1e-4, greek="delta", seed=None):
        """
        Calculate the Greeks for a barrier option using finite differences.

        Parameters:
        - option_price_fn: Function to compute the option price.
        - S0: Initial stock price.
        - K: Strike price.
        - T: Time to maturity.
        - r: Risk-free interest rate.
        - sigma: Volatility.
        - barrier: Barrier level.
        - option_type: "call" or "put".
        - barrier_type: "up-and-out" or "down-and-out".
        - delta: Dividend yield (default is 0.25).
        - n_paths: Number of Monte Carlo paths (default is 1000).
        - n_steps: Number of time steps in Monte Carlo simulation (default is 100).
        - epsilon: Small perturbation for finite difference (default is 1e-4).
        - greek: The Greek to calculate ("delta", "gamma", "vega", "theta", "rho").
        - seed: Seed for random number generation (default is None).

        Returns:
        - The calculated Greek value.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        # Base price calculation
        base_price = option_price_fn(S0, K, T, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)

        if greek == "delta":
            price_up = option_price_fn(S0 + epsilon, K, T, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            return (price_up - base_price) / epsilon

        elif greek == "gamma":
            price_up = option_price_fn(S0 + epsilon, K, T, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            price_down = option_price_fn(S0 - epsilon, K, T, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            return (price_up - 2 * base_price + price_down) / (epsilon ** 2)

        elif greek == "vega":
            price_up = option_price_fn(S0, K, T, r, sigma + epsilon, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            return (price_up - base_price) / epsilon

        elif greek == "theta":
            price_up = option_price_fn(S0, K, T - epsilon, r, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            return (base_price - price_up) / epsilon

        elif greek == "rho":
            price_up = option_price_fn(S0, K, T, r + epsilon, sigma, barrier, option_type, barrier_type, n_paths, n_steps, seed=seed)
            return (price_up - base_price) / epsilon

        else:
            raise ValueError("Unsupported Greek type. Supported types are: 'delta', 'gamma', 'vega', 'theta', 'rho'.")

