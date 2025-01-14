# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from datetime import datetime

class Pricing:
    def __init__(self, ticker = "AAPL"):
        self.ticker = ticker
        self.data = self.get_data(ticker)
        self.price = None

    def get_data(self, ticker_symbol, columns_to_extract=['lastPrice', 'strike', 'volume', 'bid', 'ask']):
        ticker = yf.Ticker(ticker_symbol)

        # Fetch the current price of the underlying asset
        current_price = ticker.history(period="1d")['Close'].iloc[-1]

        # Fetch available expiration dates for the options
        maturities = ticker.options

        # Initialize an empty list to store DataFrames for each maturity
        options_data = []

        for maturity in maturities:
            # Fetch the options chain for the current maturity
            options_chain = ticker.option_chain(maturity)

            # Extract Call and Put data, rename columns, and add maturity column
            calls = options_chain.calls[columns_to_extract].copy()
            calls.rename(columns={
                'lastPrice': 'C',
                'volume': 'C-volume',
                'bid': 'callBid',
                'ask': 'callAsk',
                'strike': 'K'
            }, inplace=True)
            calls['C bid-ask'] = calls['callAsk'] - calls['callBid']  # Calculate bid-ask spread for calls
            calls['Maturity'] = maturity
            calls['T'] = (pd.to_datetime(maturity).date() - datetime.now().date()).days / 365.25

            puts = options_chain.puts[columns_to_extract].copy()
            puts.rename(columns={
                'lastPrice': 'P',
                'volume': 'P-volume',
                'bid': 'putBid',
                'ask': 'putAsk',
                'strike': 'K'
            }, inplace=True)
            puts['P bid-ask'] = puts['putAsk'] - puts['putBid']  # Calculate bid-ask spread for puts
            puts['Maturity'] = maturity
            puts['T'] = (pd.to_datetime(maturity).date() - datetime.now().date()).days / 365.25
            # Merge Call and Put data on 'strike' and 'T'
            merged_data = pd.merge(calls, puts, on=['K', 'T','Maturity'], how='outer')
            merged_data['S'] = current_price

            # Append the merged data to the list
            options_data.append(merged_data)

        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(options_data, ignore_index=True)
        final_df.drop(['callBid', 'callAsk', 'putBid', 'putAsk'], axis=1, inplace=True)
        self.data = final_df
        return final_df
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate the Black-Scholes option price.
        :param S: Current stock price
        :param K: Option strike price
        :param T: Time to expiration (in years)
        :param r: Risk-free interest rate
        :param sigma: Volatility of the underlying stock
        :param option_type: Type of option ('call' or 'put')
        :return: Black-Scholes option price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        return option_price
    
    def implied_volatility(self, market_price, S, K, T, r, option_type):
        def objective(sigma):
            # Ensure sigma is positive (volatility must be > 0)
            if sigma <= 0:
                return np.inf
            # Compute the Black-Scholes price
            theoretical_price = self.black_scholes(S, K, T, r, sigma, option_type)
            # Return the squared error
            return (theoretical_price - market_price)**2

        # Initial guess and bounds for volatility
        initial_guess = 0.2
        bounds = [(10**-4, 5)]  # Volatility should be in a reasonable range

        # Perform the minimization
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        # Check if the minimization was successful
        if result.success:
            return result.x[0]  # Return the optimized implied volatility
        else:
            raise ValueError("Implied volatility calculation failed.")
        
    def compute_iv(self, r=0.03):
        for index, row in self.data.iterrows():
            try:
                if not np.isnan(row['P']) and (np.isnan(row['C']) or row["P bid-ask"] < row["C bid-ask"]):
                    # Use put option if C is NaN or P has a lower bid-ask spread
                    iv = self.implied_volatility(
                        S=row['S'],  # Current stock price
                        K=row['K'],  # Strike price
                        T=row['T'],  # Time to maturity
                        r=r,      # Risk-free rate
                        option_type='put',  # Option type
                        market_price=row['P']  # Observed market price
                    )

                elif not np.isnan(row['C']):
                    # Use call option if C is valid and either P is NaN or C has a lower bid-ask spread
                    iv = self.implied_volatility(
                        S=row['S'],  # Current stock price
                        K=row['K'],  # Strike price
                        T=row['T'],  # Time to maturity
                        r=r,      # Risk-free rate
                        option_type='call',  # Option type
                        market_price=row['C']  # Observed market price
                    )
                else:
                    # If both C and P are NaN, set IV to NaN
                    iv = np.nan

                # Assign the computed implied volatility to the DataFrame
                self.data.loc[index, 'IV'] = iv
            except ValueError:
                # Handle exceptions during IV computation
                self.data.loc[index, 'IV'] = np.nan
        self.data = self.data.dropna(subset=['IV']).copy()


    def compute_price(self, K, T, S,r=0.03, option_type="call"):
        """
        Compute the price of an option using the Black-Scholes formula.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate (default is 0.03).
            option_type (str): Type of option ("call" or "put"). Default is "call".

        Returns:
            float: Option price.
        """
        # Ensure the necessary columns are present
        required_columns = {"K", "T", "IV"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"The dataframe must contain the columns: {required_columns}")


        # Case 1: Direct match
        if K in self.data["K"].values and T in self.data["T"].values:
            row = self.data[(self.data["K"] == K) & (self.data["T"] == T)]
            sigma = row["IV"].values[0]
            return self.black_scholes(S, K, T, r, sigma, option_type), sigma

        try:
            # Step 1: Find neighboring K values
            K_lower = self.data[self.data["K"] <= K]["K"].max()
            K_upper = self.data[self.data["K"] > K]["K"].min()

            if pd.isna(K_lower) or pd.isna(K_upper):
                raise ValueError("K is out of bounds for interpolation.")

            # Step 2: Interpolate across T for each K
            sigma_K_lower = self.interpolate_sigma(self.data[self.data["K"] == K_lower], T)
            sigma_K_upper = self.interpolate_sigma(self.data[self.data["K"] == K_upper], T)

            # Step 3: Interpolate across K
            sigma = sigma_K_lower + (sigma_K_upper - sigma_K_lower) * (K - K_lower) / (K_upper - K_lower)

            # Step 4: Price the option
            return self.black_scholes(S, K, T, r, sigma, option_type), sigma

        except Exception as e:
            raise ValueError(f"Interpolation failed: {e}")

    def interpolate_sigma(self, data, T):
        """Interpolate sigma for a given T within a specific K."""
        T_lower = data[data["T"] <= T]["T"].max()
        T_upper = data[data["T"] > T]["T"].min()

        if pd.isna(T_lower) or pd.isna(T_upper):
            raise ValueError("T is out of bounds for interpolation.")

        sigma_lower = data[data["T"] == T_lower]["IV"].values[0]
        sigma_upper = data[data["T"] == T_upper]["IV"].values[0]
        return sigma_lower + (sigma_upper - sigma_lower) * (T - T_lower) / (T_upper - T_lower)

    def delta_greek(self, K, T, S, sigma, r=0.03, option_type="call"):
        """
        Calculate the Delta Greek for an option.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            S (float): Current stock price.
            sigma (float): Implied volatility.
            r (float): Risk-free interest rate. Default is 0.03.
            option_type (str): "call" or "put". Default is "call".

        Returns:
            float: Delta of the option.
        """
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be greater than 0.")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def gamma_greek(self, K, T, S, sigma, r=0.03):
        """
        Calculate the Gamma Greek for an option.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            S (float): Current stock price.
            sigma (float): Implied volatility.
            r (float): Risk-free interest rate. Default is 0.03.

        Returns:
            float: Gamma of the option.
        """
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be greater than 0.")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega_greek(self, K, T, S, sigma, r=0.03):
        """
        Calculate the Vega Greek for an option.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            S (float): Current stock price.
            sigma (float): Implied volatility.
            r (float): Risk-free interest rate. Default is 0.03.

        Returns:
            float: Vega of the option.
        """
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be greater than 0.")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def theta_greek(self, K, T, S, sigma, r=0.03, option_type="call"):
        """
        Calculate the Theta Greek for an option.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            S (float): Current stock price.
            sigma (float): Implied volatility.
            r (float): Risk-free interest rate. Default is 0.03.
            option_type (str): "call" or "put". Default is "call".

        Returns:
            float: Theta of the option.
        """
        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be greater than 0.")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    def compute_greeks(self, K, T, S, sigma, r=0.03, option_type="call"):
        """
        Compute the price and greeks of an option using the Black-Scholes formula.

        Parameters:
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Risk-free interest rate (default is 0.03).
            option_type (str): Type of option ("call" or "put"). Default is "call".

        Returns:
            dict: Option price and greeks.
        """
        try:
            # Compute the greeks
            delta = self.delta_greek(K, T, S, r, sigma, option_type)
            gamma = self.gamma_greek(K, T, S, r, sigma)
            vega = self.vega_greek(K, T, S, r, sigma)
            theta = self.theta_greek(K, T, S, r, sigma, option_type)

            return {
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta
            }

        except Exception as e:
            raise ValueError(f"Greek computation failed: {e}")
