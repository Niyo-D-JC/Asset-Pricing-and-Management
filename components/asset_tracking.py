# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class IndexReplication:
    def __init__(self, index_ticker, tickers, start_date, end_date, frequency='W-FRI'):
        """
        Initialize the index replication class.

        Args:
            index_ticker (str): Ticker symbol for the index.
            tickers (list): List of ticker symbols for portfolio components.
            start_date (str): Start date for data collection.
            end_date (str): End date for data collection.
            frequency (str): Frequency for data resampling ('W-FRI' for weekly, 'M' for monthly).
        """
        self.index_ticker = index_ticker
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.portfolio_data = None
        self.benchmark_data = None
        self.weights = None

    def get_data(self):
        """
        Fetch historical data for portfolio and benchmark.
        """
        portfolio_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        benchmark_data = yf.download(self.index_ticker, start=self.start_date, end=self.end_date)['Close']

        # Resample data to the desired frequency
        portfolio_data = portfolio_data.resample(self.frequency).last()
        benchmark_data = benchmark_data.resample(self.frequency).last()

        self.portfolio_data = portfolio_data.dropna()
        self.benchmark_data = benchmark_data.dropna()

    def calculate_tracking_error(self, weights, benchmark_returns, portfolio_returns, rho_b_p=1,period=52):
        """
        Compute the tracking error.
        """
        covariance_matrix = portfolio_returns.cov().to_numpy() * period
        weights = np.array(weights).reshape(-1, 1)
        sigma_portfolio = weights.T @ covariance_matrix @ weights
        sigma_benchmark = benchmark_returns.var() * period
        # Tracking error formula
        TE = np.sqrt(
            sigma_portfolio.item()
            + sigma_benchmark
            - 2 * rho_b_p * np.sqrt(sigma_portfolio.item()) * np.sqrt(sigma_benchmark)
        )
        return TE

    def optimize_tracking_error(self, period=52, tol=1e-6):
        """
        Optimize portfolio weights to minimize tracking error.
        """
        benchmark_returns = np.log(self.benchmark_data / self.benchmark_data.shift(1)).dropna()
        portfolio_returns = np.log(self.portfolio_data / self.portfolio_data.shift(1)).dropna()

        n_assets = portfolio_returns.shape[1]

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial weights
        initial_weights = np.ones(n_assets) / n_assets

        # Minimize tracking error
        result = minimize(
            fun=lambda w: self.calculate_tracking_error(w, benchmark_returns, portfolio_returns, period=period),
            x0=initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            tol=tol
        )

        if result.success:
            self.weights = {ticker: weight for ticker, weight in zip(portfolio_returns.columns, result.x)}
            return self.weights
        else:
            raise ValueError(f"Optimization failed: {result.message}")

    def compute_annualized_returns(self):
        """
        Compute the annualized returns for the portfolio and benchmark.
        """
        if self.weights is None:
            raise ValueError("Weights not optimized. Call `optimize_tracking_error` first.")

        weights_array = np.array(list(self.weights.values()))
        benchmark_returns = np.log(self.benchmark_data / self.benchmark_data.shift(1)).dropna()
        portfolio_returns = np.log(self.portfolio_data / self.portfolio_data.shift(1)).dropna()
        portfolio_total_returns = portfolio_returns @ weights_array

        annualized_benchmark_return = (1 + benchmark_returns).cumprod() - 1
        annualized_portfolio_return = (1 + portfolio_total_returns).cumprod() - 1

        return annualized_benchmark_return, annualized_portfolio_return

