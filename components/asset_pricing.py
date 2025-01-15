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

    def get_data(self, ticker_symbol):
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        data = []
        for exp in expirations:
            options_chain = ticker.option_chain(exp)
            calls = options_chain.calls
            puts = options_chain.puts

            # Créer un DataFrame avec un index basé sur les strikes
            calls = calls.set_index("strike")
            puts = puts.set_index("strike")

            # Combiner les données des calls et puts sur le même strike
            combined = calls.join(puts, how="outer", lsuffix="_call", rsuffix="_put")
            for strike, row in combined.iterrows():
                data.append([
                    strike,
                    exp,
                    row.get("lastPrice_call", None),
                    row.get("lastPrice_put", None),
                    row.get("volume_call", 0),
                    row.get("volume_put", 0),
                    (row.get("bid_call", None), row.get("ask_call", None)),
                    (row.get("bid_put", None), row.get("ask_put", None))
                ])
        options_table = pd.DataFrame(data, columns=["K", "T", "C", "P", "Volume Call", "Volume Put", "Bid/Ask Call", "Bid/Ask Put"])

        options_table["T"] = pd.to_datetime(options_table["T"]) - pd.Timestamp.now()
        options_table["T"] = options_table["T"].dt.days / 365.0  # Convertir en années

        # Supprimer les lignes sans prix de call et de put
        # options_table = options_table.dropna(subset=["C", "P"], how="any").sort_values(by=["K", "T"])
        # Remplacer les NaN dans les colonnes "Volume Call" et "Volume Put" par 0
        options_table["Volume Call"] = options_table["Volume Call"].fillna(0)
        options_table["Volume Put"] = options_table["Volume Put"].fillna(0)

        return options_table
    
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
    
    def implied_volatility(self, market_price, S, K, T, r, option_type, tol=1e-3, max_iter=100):

        sigma = 0.4  # Initial guess pour la volatilité
        for i in range(max_iter):
            # Calculer le prix du call avec la volatilité courante
            call_price = self.black_scholes(S, K, T, r, sigma, option_type=option_type)
             

            # Calculer Vega
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)

            # Mettre à jour sigma
            price_diff = call_price - market_price  # Erreur entre le prix calculé et le prix de marché
            if abs(price_diff) < tol:
                return sigma  # Convergence atteinte
            sigma -= price_diff / vega  # Méthode de Newton-Raphson

        return np.nan
        
    def compute_iv(self, r=0.03):
        option_type = None
        implied_vols = []
        for _, row in self.data.iterrows():
            K = row["K"]
            T = row["T"]
            call_price = row["C"]
            put_price = row["P"]
            volume_call = row["Volume Call"]
            volume_put = row["Volume Put"]

            # Choisir le type d'option et le prix en fonction du volume
            if volume_call >= volume_put:
                option_price = call_price
                option_type = "call"
            else:
                option_price = put_price
                option_type = "put"

            # Calculer la volatilité implicite
            
            implied_vol = self.implied_volatility(option_price, self.price, K, T, r,option_type)
            implied_vols.append(implied_vol)

        # Ajouter les résultats au DataFrame
        self.data["IV"] = implied_vols
        self.data['Max_vol'] = np.maximum(self.data['Volume Call'], self.data['Volume Put'])
        self.data[(self.data['IV'] > 0) & (self.data['Max_vol'] > 5)]

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
        vol_target = griddata(points, volatilities, (K_target, T_target), method='cubic')

        if np.isnan(vol_target):
            raise ValueError("L'interpolation a échoué. Vérifiez les données disponibles.")

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
        self.data["Rho"] = np.nan

        # Calculer les Grecques pour chaque ligne
        for idx, row in self.data.iterrows():
            K = row["K"]
            T = row["T"]
            vol = row["Implied Vol"]

            if T <= 0 or vol <= 0:
                continue  # Ignorer les cas non valides

            # Calcul des paramètres d1 et d2
            d1 = (np.log(self.price / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)

            # Calcul des Grecques
            if option_type == "call":
                delta = norm.cdf(d1)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == "put":
                delta = norm.cdf(d1) - 1
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
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
            self.data.at[idx, "Rho"] = rho