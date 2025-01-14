# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from datetime import datetime

class Management:
    def __init__(self, assets = []):
        self.assets = assets
        self.returns , self.data = self.get_returns(assets)
        self.mu , self.sigma = [], None

    def get_returns(self, assets):
        data = yf.download(start="2010-01-01", group_by="ticker")

        # Initialisation d'un DataFrame pour les rendements
        returns = pd.DataFrame()

        # Calcul des log-rendements négatifs pour chaque actif
        for asset in assets:
            try:
                # Vérifie si la colonne 'Close' est disponible
                if 'Close' in data[asset]:
                    # Log-rendements négatifs
                    returns[asset] = np.log(data[asset]['Close'] / data[asset]['Close'].shift(1))
            except KeyError:
                print(f"Données manquantes pour {asset}")
        returns = returns.dropna()
        return returns, data
    
    def get_parameters(self, freq = "day"):
        if freq == "day":
            
            for asset in self.assets:
                first_value = self.data[asset]['Close'].dropna().iloc[0]
                last_value = self.data[asset]['Close'].dropna().iloc[-1]
                r_p = np.log(last_value / first_value)
                mu_an = (r_p + 1)**(1/10) - 1
                self.mu.append(mu_an)

        self.sigma = self.returns.cov().to_numpy()
        self.sigma = 252*self.sigma
        return self.mu, self.sigma
    
    def portfolio_variance(self, weights):
        return weights.T @ self.sigma @ weights
    
    def weight_sum_constraint(self, weights):
        return np.sum(weights) - 1

    def target_return_constraint(self, weights, mu_target):
        return weights.T @ self.mu - mu_target
    
    def efficient_portfolio(self, mu_target, range_=(-0.1, None)):
        # Initialisation des poids
        n_assets = len(self.mu)
        init_weights = np.ones(n_assets) / n_assets
        init_weights = init_weights

        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': self.weight_sum_constraint},
            {'type': 'ineq', 'fun': lambda w: self.target_return_constraint(w, mu_target)}
        ]

        # Contraintes de positivité (pas de vente à découvert)
        bounds = [range_ for _ in range(n_assets)]

        result = minimize(self.portfolio_variance, init_weights, bounds= bounds, method='SLSQP', constraints=constraints)
        if result.success:
            optimal_weights = result.x
            portfolio_volatility = np.sqrt(self.portfolio_variance(optimal_weights))
            return portfolio_volatility, optimal_weights
        else :
            return None,None