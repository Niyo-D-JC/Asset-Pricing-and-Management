import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import date

class IndexReplication:
    def __init__(self, index_ticker, component_tickers, start_date, end_date, period=52, monthly=False):
        self.index_ticker = index_ticker
        self.component_tickers = component_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.monthly = monthly
        self.weights_history = []
        self.data = None
        self.portfolio_data = None
        self.benchmark_data = None

    def get_data(self):
        """
        Fetch historical data for the index and its components.
        """
        data = yf.download(self.component_tickers, start=self.start_date, end=self.end_date)['Close']
        index = yf.download(self.index_ticker, start=self.start_date, end=self.end_date)['Close']

        if self.monthly:
            data = data.resample('M').last()
            index = index.resample('M').last()
        else:
            data = data.resample('W-FRI').last()
            index = index.resample('W-FRI').last()

        self.data = {"portfolio_data": data, "benchmark_data": index}
        self.portfolio_data = data
        self.benchmark_data = index
        return self.data

    @staticmethod
    def calculate_tracking_error(weights, benchmark_returns, portfolio_returns, rho_b_p=1, period=52):
        """
        Calculate tracking error between portfolio and benchmark.
        """
        covariance_matrix = portfolio_returns.cov().to_numpy() * np.sqrt(period)
        sigma_portfolio = weights.T @ covariance_matrix @ weights
        sigma_benchmark = benchmark_returns.var() * np.sqrt(period)

        return np.sqrt(sigma_portfolio + sigma_benchmark - 2 * rho_b_p * np.sqrt(sigma_portfolio) * np.sqrt(sigma_benchmark))

    @staticmethod
    def annualized_return(returns, period=52):
        """
        Computes the annualized return.
        """
        mean_return = np.mean(returns)
        cumulative_return = (1 + mean_return) ** period
        return cumulative_return - 1

    def optimize_tracking_error(self, train_benchmark, train_portfolio, tol=1e-6):
        """
        Optimize portfolio weights to minimize tracking error.
        """
        benchmark_returns = np.log(train_benchmark / train_benchmark.shift(1)).dropna()
        portfolio_returns = np.log(train_portfolio / train_portfolio.shift(1)).dropna()

        n_assets = portfolio_returns.shape[1]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            fun=lambda w: self.calculate_tracking_error(w, benchmark_returns, portfolio_returns, period=self.period),
            x0=initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            tol=tol
        )

        if result.success:
            return {ticker: weight for ticker, weight in zip(portfolio_returns.columns, result.x)}
        else:
            print("Message:", result.message)
            raise ValueError("L'optimisation du Tracking Error a échoué.")

    def run_backtest(self):
        """
        Perform backtesting on the portfolio.
        """
        # Add Year column for grouping
        self.benchmark_data["Year"] = self.benchmark_data.index.year
        self.portfolio_data["Year"] = self.portfolio_data.index.year

        tracking_results = []
        all_portfolio_returns = []
        all_benchmark_returns = []

        years = sorted(self.benchmark_data["Year"].unique())[:-1]

        for year in years:
            train_benchmark = self.benchmark_data[self.benchmark_data["Year"] == year].drop(columns=["Year"])
            train_portfolio = self.portfolio_data[self.portfolio_data["Year"] == year].drop(columns=["Year"])

            test_benchmark = self.benchmark_data[self.benchmark_data["Year"] == year + 1].drop(columns=["Year"])
            test_portfolio = self.portfolio_data[self.portfolio_data["Year"] == year + 1].drop(columns=["Year"])

            optimized_weights = self.optimize_tracking_error(train_benchmark, train_portfolio)

            weights = np.array(list(optimized_weights.values()))
            self.weights_history.append(optimized_weights)
            
            test_returns = np.log(test_benchmark / test_benchmark.shift(1)).dropna()
            portfolio_test_returns = np.log(test_portfolio / test_portfolio.shift(1)).dropna()
            portfolio_total_returns = portfolio_test_returns @ weights

            tracking_error = self.calculate_tracking_error(
                weights, test_returns, portfolio_test_returns, period=self.period
            )

            tracking_results.append({"Year": year + 1, "Tracking Error": tracking_error})
            all_portfolio_returns.append(portfolio_total_returns)
            all_benchmark_returns.append(test_returns)

        all_portfolio_returns = pd.concat(all_portfolio_returns)
        all_benchmark_returns = pd.concat(all_benchmark_returns)

        annualized_portfolio_return = (1 + all_portfolio_returns).cumprod() - 1
        annualized_benchmark_return = (1 + all_benchmark_returns).cumprod() - 1

        tracking_df = pd.DataFrame(tracking_results)

        return tracking_df, annualized_portfolio_return, annualized_benchmark_return
