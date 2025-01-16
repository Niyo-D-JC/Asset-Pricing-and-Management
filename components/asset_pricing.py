# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

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


    def price_option_by_interpolation(self, K_target, T_target, S, r = 0.03, option_type="call"):
        """
        Pricer une option hors marché en interpolant la volatilité implicite.

        Paramètres :
        - df : DataFrame contenant ["K", "T", "Implied Vol"]
        - S : Prix actuel du sous-jacent
        - r : Taux sans risque
        - K_target : Strike de l'option à pricer
        - T_target : Maturité de l'option à pricer
        - option_type : Type d'option ("call" ou "put")

        Retourne :
        - Prix interpolé de l'option
        """
        # Extraire les données pour interpolation
        strikes = self.data["K"].values
        maturities = self.data["T"].values
        volatilities = self.data["IV"].values

        # Interpoler la volatilité implicite à (K_target, T_target)
        points = np.column_stack((strikes, maturities))
        vol_target = griddata(points, volatilities, (K_target, T_target))

        if np.isnan(vol_target):
            vol_target = griddata(points, volatilities, (K_target, T_target), method='nearest')

        # Calculer le prix de l'option avec Black-Scholes
        d1 = (np.log(S / K_target) + (r + 0.5 * vol_target**2) * T_target) / (vol_target * np.sqrt(T_target))
        d2 = d1 - vol_target * np.sqrt(T_target)

        if option_type == "call":
            price = S * norm.cdf(d1) - K_target * np.exp(-r * T_target) * norm.cdf(d2)
        elif option_type == "put":
            price = K_target * np.exp(-r * T_target) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Type d'option invalide. Choisissez 'call' ou 'put'.")

        return price, vol_target


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
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def gamma_greek(self, K, T, S, sigma, r=0.03, option_type="call"):
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
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega_greek(self, K, T, S, sigma, r=0.03, option_type="call"):
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

    def calculate_greeks(self, r, option_type="call"):
        """
        Calculer les Grecques (\Delta, \Gamma, \Theta, \Vega, \Rho) pour chaque combinaison K et T.

        Paramètres :
        - df : DataFrame contenant ["K", "T", "Implied Vol"]
        - S : Prix actuel du sous-jacent
        - r : Taux sans risque
        - option_type : Type d'option ("call" ou "put")

        Retourne :
        - DataFrame enrichi avec ["Delta", "Gamma", "Theta", "Vega", "Rho"]
        """
        # Initialiser les colonnes pour les Grecques
        self.data["Delta"] = np.nan
        self.data["Gamma"] = np.nan
        self.data["Theta"] = np.nan
        self.data["Vega"] = np.nan

        # Calculer les Grecques pour chaque ligne
        for idx, row in self.data.iterrows():
            K = row["K"]
            T = row["T"]
            vol = row["IV"]

            if T <= 0 or vol <= 0:
                continue  # Ignorer les cas non valides

            # Calcul des paramètres d1 et d2
            d1 = (np.log(self.price / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)

            # Calcul des Grecques
            if option_type == "call":
                delta = norm.cdf(d1)
            elif option_type == "put":
                delta = norm.cdf(d1) - 1
            else:
                raise ValueError("Type d'option invalide. Choisissez 'call' ou 'put'.")

            gamma = norm.pdf(d1) / (self.price * vol * np.sqrt(T))
            theta = (-self.price * norm.pdf(d1) * vol / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2))
            vega = self.price * norm.pdf(d1) * np.sqrt(T)

            # Ajouter les valeurs au DataFrame
            self.data.at[idx, "Delta"] = delta
            self.data.at[idx, "Gamma"] = gamma
            self.data.at[idx, "Theta"] = theta
            self.data.at[idx, "Vega"] = vega
