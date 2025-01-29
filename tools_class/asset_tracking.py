import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

class IndexReplication:
    def __init__(self, index_ticker, component_tickers, start_date, end_date, monthly=False):
        self.index_ticker = index_ticker
        self.component_tickers = component_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.monthly = monthly
        self.period = 52
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

        data = data.resample('W-FRI').last()
        index = index.resample('W-FRI').last()

        self.data = {"portfolio_data": data, "benchmark_data": index}
        self.portfolio_data = data
        self.benchmark_data = index
        return self.data
    
    def get_sub_data(self, start_date, end_date):
        """
        Fetch historical data for the index and its components.
        """
        data = self.data["portfolio_data"].loc[
            (self.data["portfolio_data"].index > start_date) & 
            (self.data["portfolio_data"].index <= end_date)
        ]
        index = self.data["benchmark_data"].loc[
            (self.data["benchmark_data"].index > start_date) & 
            (self.data["benchmark_data"].index <= end_date)
        ]

        data = data.resample('W-FRI').last()
        index = index.resample('W-FRI').last()

        self.portfolio_data = data
        self.benchmark_data = index

    @staticmethod
    def calculate_tracking_error(weights, benchmark_returns, portfolio_returns, rho_b_p=1, period=52):
        """
        Calculate tracking error between portfolio and benchmark.
        """
        covariance_matrix = np.cov(portfolio_returns, rowvar=False) * period
        var_portfolio = weights.T @ covariance_matrix @ weights
        var_benchmark = (np.var(benchmark_returns, axis=0) * period).iloc[0]
        
        # Minimize tracking error formula is like minimizing the following function
        term = float(var_portfolio + var_benchmark - 2 * rho_b_p * np.sqrt(var_portfolio) * np.sqrt(var_benchmark))
        if term<0:
            term = 1
        
        return np.sqrt(term)
        
        #return np.sqrt(np.mean(diff ** 2))
    
    def optimize_tracking_error(self, train_benchmark, train_portfolio, tol=1e-6, max_assets = 40):
        """
        Optimize portfolio weights to minimize tracking error.

        Returns:
        - tracking_df: Calculated tracking error for the optimized weights.
        """
        benchmark_returns = np.log(train_benchmark / train_benchmark.shift(1)).dropna()
        portfolio_returns = np.log(train_portfolio / train_portfolio.shift(1)).dropna()

        n_assets = portfolio_returns.shape[1]
        
        # Initial weights (equal weighting)
        initial_weights = np.ones(n_assets) / n_assets

        # Bounds for weights (between 0 and 1)
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Objective function with penalty for exceeding max_assets
        def objective_with_penalty(weights):
            binary_selection = (weights > 0).astype(int)  # Consider weights > 0 as selected
            num_selected_assets = np.sum(binary_selection)

            # Penalty for selecting more than max_assets
            penalty = max(0, num_selected_assets - max_assets)/n_assets

            #bool_pen = 1-(num_selected_assets == max_assets).astype(int) 

            # Calculate the tracking error
            tracking_error = self.calculate_tracking_error(weights, benchmark_returns, portfolio_returns, period=self.period)

            return tracking_error + penalty #+ bool_pen

        # Constraint: weights must sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Minimize the objective
        result = minimize(
            fun=objective_with_penalty,
            x0=initial_weights,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            tol=tol
        )

        if result.success:
            w_opt = result.x
            w_opt[w_opt<= 0] = 0
            w_opt = w_opt/w_opt.sum()
            return {ticker: weight for ticker, weight in zip(portfolio_returns.columns, result.x)}
        else:
            return {ticker: weight for ticker, weight in zip(portfolio_returns.columns, initial_weights)}
            
    def run_backtest(self, max_assets = 40):
        """
        Perform backtesting with sliding 1-year training periods and flexible test periods.
        """
        # Ensure benchmark and portfolio data are in datetime format
        self.benchmark_data.index = pd.to_datetime(self.benchmark_data.index)
        self.portfolio_data.index = pd.to_datetime(self.portfolio_data.index)
    
        tracking_results = []
        all_portfolio_returns = []
        all_benchmark_returns = []
    
        # Define rebalancing frequency
        if self.monthly:
            step = "1ME"  # Slide training by 1 month
            test_freq = "ME"
        else:
            step = "12ME"  # Slide training by 1 year
            test_freq = "YE"

        # Initialize training start date
        start_date = self.benchmark_data.index.min()
    
        while True:
            # Define training period: 1 year
            train_start_date = start_date
            train_end_date = train_start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    
            # Define test period: immediately after training
            test_start_date = train_end_date + pd.Timedelta(days=1)
            test_end_date = test_start_date + pd.tseries.frequencies.to_offset(test_freq) - pd.Timedelta(days=1)
    
            # Ensure test period is within available data range
            if test_end_date > self.benchmark_data.index.max():
                break
            
            # Select training and test data
            train_benchmark = self.benchmark_data.loc[train_start_date:train_end_date]
            train_portfolio = self.portfolio_data.loc[train_start_date:train_end_date]
            test_benchmark = self.benchmark_data.loc[test_start_date:test_end_date]
            test_portfolio = self.portfolio_data.loc[test_start_date:test_end_date]
    
            # Handle empty training or testing data
            if train_benchmark.empty or test_benchmark.empty:
                break
            
            # Optimize weights using the training data
            optimized_weights = self.optimize_tracking_error(train_benchmark, train_portfolio, max_assets = max_assets)
            
            # Convert weights to array and store them
            weights = np.array(list(optimized_weights.values()))

            self.weights_history.append(optimized_weights)
    
            # Compute test period returns
            test_returns = np.log(test_benchmark / test_benchmark.shift(1)).dropna()
            portfolio_test_returns = np.log(test_portfolio / test_portfolio.shift(1)).dropna()
    
            # Ensure alignment of weights and test portfolio returns
            portfolio_test_returns = portfolio_test_returns[optimized_weights.keys()]
            portfolio_total_returns = portfolio_test_returns.to_numpy() @ weights
    
            # Calculate tracking error for the test period
            tracking_error = self.calculate_tracking_error(
                weights, test_returns, portfolio_test_returns, period=self.period
            )
    
            # Store tracking error and period
            tracking_results.append({
                "Period": test_start_date.strftime("%Y-%m-%d"),
                "Tracking Error": tracking_error
            })
    
            # Append the returns to the list
            all_portfolio_returns.append(pd.Series(portfolio_total_returns, index=test_returns.index))
            all_benchmark_returns.append(test_returns)
    
            # Slide the training period
            start_date += pd.tseries.frequencies.to_offset(step)
    
        # Combine all returns into single Series
        all_portfolio_returns = pd.concat(all_portfolio_returns, axis=0)
        all_benchmark_returns = pd.concat(all_benchmark_returns, axis=0)
    
        # Calculate annualized cumulative returns
        annualized_portfolio_return = (1 + all_portfolio_returns).cumprod() - 1
        annualized_benchmark_return = (1 + all_benchmark_returns).cumprod() - 1
    
        # Create tracking results DataFrame
        tracking_df = pd.DataFrame(tracking_results)
    
        return tracking_df, annualized_portfolio_return, annualized_benchmark_return
        