# Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import numpy as np

class Management:
    def __init__(self, assets = []):
        self.assets = assets
        self.returns , self.data = self.get_returns(assets)
        self.mu , self.sigma = [], None

    def get_returns(self, assets):
        data = yf.download(assets, start="2010-01-01", group_by="ticker")

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
    
    def get_parameters(self, freq="day"):
        valid_assets = []  # Liste pour les actifs valides
        self.mu = []  # Liste pour stocker les mu
        self.sigma = None  # Matrice de covariance

        if freq == "day":
            for asset in self.assets:
                try:
                    first_value = float(self.data[asset]['Close'].dropna().iloc[0])
                    last_value = float(self.data[asset]['Close'].dropna().iloc[-1])
                    r_p = np.log(last_value / first_value)

                    # Vérifie si r_p > -1
                    if r_p > -1:
                        valid_assets.append(asset)
                        mu_an = (r_p + 1)**(1/10) - 1  # Calcul du rendement annualisé
                        self.mu.append(mu_an)
                    else:
                        print(f"The asset {asset} is excluded because r_p = {r_p} < -1")
                except KeyError:
                    print(f"Missing data for {asset}")

        # Recalcul de sigma avec les actifs valides
        if valid_assets:
            filtered_returns = self.returns[valid_assets]
            self.sigma = filtered_returns.cov().to_numpy()
            self.sigma = 252 * self.sigma  # Ajustement annuel

            self.assets = valid_assets
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