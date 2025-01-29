import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math

class Zero_Coupons:
    def __init__(self, pas = 0.5):
        self.pas = pas  # Pas utilisé pour générer les maturités cibles (par défaut 0.5).

    def interpolation_loglin(self, t1, t2, r1, r2, t):
        """
        Interpole log-linéairement un taux de rendement entre deux points.

        t1, t2 : deux points de temps (en général t1 < t2)
        r1, r2 : taux de rendement correspondant aux temps t1 et t2
        t : le temps pour lequel on souhaite interpoler le taux de rendement
        """
        ln_1_plus_r1 = np.log(1 + r1)
        ln_1_plus_r2 = np.log(1 + r2)
        ln_1_plus_rt = ln_1_plus_r1 + (t - t1) / (t2 - t1) * (ln_1_plus_r2 - ln_1_plus_r1)
        r = np.exp(ln_1_plus_rt) - 1
        return r
    
    def nelson_siegel(self, tau, beta0, beta1, beta2, lamb):
        """
        tau : maturité (en années)
        beta0, beta1, beta2 : paramètres Nelson-Siegel
        lamb : paramètre de vitesse d'ajustement
        """
        term1 = beta0
        term2 = beta1 * (1 - np.exp(-tau / lamb)) / (tau / lamb)
        term3 = beta2 * ((1 - np.exp(-tau / lamb)) / (tau / lamb) - np.exp(-tau / lamb))
        return term1 + term2 + term3


    def objective_function_nelson_siegel(self, params, tau, rates):
        #rates : taux zéro-coupon observés
        beta0, beta1, beta2, lamb = params
        fitted_rates = self.nelson_siegel(tau, beta0, beta1, beta2, lamb)
        return np.sum((fitted_rates - rates) ** 2)  # Somme des erreurs quadratiques

    def interpoler_taux_by_loglin(self, maturites, taux_zc):
        """
        Interpole les taux zéro-coupon pour des maturités cibles.

        Paramètre :
        ----------
        maturites, taux_zc : du dataframe

        Retourne :
        -------
        np.array
            Liste des taux interpolés.
        """
        maturites_cibles = np.arange(min(maturites), max(maturites) + self.pas, self.pas)
        taux_interpolés = []

        for t in maturites_cibles:
            if t in maturites.values:
                taux_interpolés.append(taux_zc[maturites == t].values[0])
            else:
                idx_sup = np.searchsorted(maturites, t)
                t1, t2 = maturites[idx_sup - 1], maturites[idx_sup]
                r1, r2 = taux_zc[idx_sup - 1], taux_zc[idx_sup]
                taux_interpolés.append(self.interpolation_loglin(t1, t2, r1, r2, t))

        return np.array(taux_interpolés)
    
    def interpoler_taux_by_nelson_siegel(self, maturites, taux_zc):
        """
        Interpole les taux zéro-coupon pour des maturités cibles.

        Paramètre :
        ----------
        maturites, taux_zc : du dataframe

        Retourne :
        -------
        np.array
            Liste des taux interpolés.
        """
        initial_params = [2.5, -1.0, 1.0, 2.0]  # (beta0, beta1, beta2, lambda)

        # Ajustement des paramètres
        result = minimize(self.objective_function_nelson_siegel, initial_params, args=(maturites, taux_zc), method='L-BFGS-B')

        beta0_opt, beta1_opt, beta2_opt, lamb_opt = result.x

        tau_fine = np.linspace(0.1, max(maturites), 100)  # Maturités entre 0.1 an et la maturité maximale
        fitted_rates = self.nelson_siegel(tau_fine, beta0_opt, beta1_opt, beta2_opt, lamb_opt)

        return fitted_rates
    
    def encadrer_reel(self, nombre):
        """
        Retourne les deux entiers qui encadrent un nombre réel.

        Arguments :
        - nombre : un nombre réel.

        Retour :
        - (entier_inf, entier_sup) : un tuple contenant l'entier inférieur et l'entier supérieur.
        """
        entier_inf = math.floor(nombre)  # L'entier immédiatement inférieur
        entier_sup = math.ceil(nombre)  # L'entier immédiatement supérieur
        return entier_inf, entier_sup
    

    def pricer_obligation_flexible(self, coupon, principal, amplitude_echeances ,maturite, taux_zero_coupons):
        """
        Cette fonction a pour but de pricer une une obligation avec ou sans coupons, au cas où on a une  des coupons,
        il est possible de spécifier l'amplitude des échéances qui n'est pas nécéssairement annuelle.

        Elle prend en entrée :

        coupon : la valeur des coupons
        principal : la valeur du principal
        amplitude_échéances : l'amplitude des échéances
        maturité : la maturité
        taux_zero_coupons : les taux zeros coupons
        """

        n = int(maturite/amplitude_echeances) # le nombre d'échéances
        prix = 0
        if coupon == 0:
            if maturite <= 1:
                return principal * np.exp(-(taux_zero_coupons.iloc[0]/100) * maturite)
        if maturite >1 :
            if maturite - round(maturite) == 0:
                return principal * np.exp(-(taux_zero_coupons.iloc[maturite - 1]/100) * maturite)
            if maturite - round(maturite) != 0:
                mat_low , mat_up = self.encadrer_reel(maturite)
                r_low = taux_zero_coupons.iloc[mat_low - 1]
                r_up = taux_zero_coupons.iloc[mat_up - 1]
                r_mat = self.interpolation_loglin(t1=mat_low, t2=mat_up, r1=r_low, r2=r_up, t=maturite)
                return principal * np.exp(-r_mat * maturite)
        if coupon > 0:
            for t in range(1, n + 1):
                # Flux de coupon pour chaque année, y compris le principal à la fin
                flux_annee = coupon if t < n else principal
            # on actualise  les flux avec le taux zéro-coupon correspondant
                if round(t*amplitude_echeances) == t*amplitude_echeances:
                    flux_act = flux_annee * np.exp(-(taux_zero_coupons.iloc[int(t*amplitude_echeances) - 1] / 100) * t*amplitude_echeances)
                else:
                    period_low, period_up = self.encadrer_reel(t*amplitude_echeances)
                    if period_low < 0:
                        r_low = taux_zero_coupons[0]
                    else:
                        r_low = taux_zero_coupons.iloc[period_low - 1]
                    r_up = taux_zero_coupons.iloc[period_up - 1]
                    r_t = self.interpolation_loglin(t1=period_low, t2=period_up, r1=r_low, r2=r_up, t=t*amplitude_echeances)
                    flux_act = flux_annee * np.exp(-(r_t* t*amplitude_echeances))
                prix = prix + flux_act
        return prix

    def calculate_fixed_swap_rate_flexible(self, tx_zero_coupon,discount_factors, delta,maturities):
        """
        Calcule le taux fixe (K) qui annule la valeur du swap payeur en utilisant les données de taux ZC.

        Paramètres :
        - tx_zero_coupon : DataFrame contenant les données (Maturité, Taux ZC, Coefficient d'actualisation).
        - delta : l'écart entre les périodes Ti-1 et Ti. (non nécéssairement annuel)

        Retour :
        - K : Le taux fixe qui égalise la valeur du swap à 0.
        """
        # Extraire les coefficients d'actualisation (B(t, T)) depuis la table
        #discount_factors = tx_zero_coupon["Coef"]

        # Initialisation des sommes pour le numérateur et le dénominateur
        numérateur = 0
        dénominateur = 0
        n = int(maturities/delta)

        for i in range(1, n+1):
            # Prix de l'obligation zéro-coupon pour T_i et T_{i-1}
            Ti = i*delta
            Ti_plus_1 = (i+1)*delta
            if Ti <=1:
                B_t_Ti = discount_factors.iloc[0]
            if Ti >1:
                if Ti % 1 == 0:
                    B_t_Ti = discount_factors.iloc[int(Ti)]
            else:
                Ti_low, Ti_up = self.encadrer_reel(Ti)
                r_Ti =  self.interpolation_loglin(t1=Ti_low, t2=Ti_up , r1=tx_zero_coupon.iloc[int(Ti_low)], r2=tx_zero_coupon.iloc[int(Ti_up)], t=Ti)
                B_t_Ti = np.exp((-r_Ti/100) * Ti)

            if Ti_plus_1 <=1:
                B_t_Ti_plus_1 = discount_factors.iloc[0]
            if Ti_plus_1 >1:
                if Ti_plus_1 % 1 == 0:
                    B_t_Ti_plus_1 = discount_factors.iloc[int(Ti_plus_1)]
                else:
                    Ti_plus_1_low, Ti_plus_1_up = self.encadrer_reel(Ti_plus_1)
                    r_Ti_plus_1 = self.interpolation_loglin(t1=Ti_plus_1_low, t2=Ti_plus_1_up , r1=tx_zero_coupon.iloc[int(Ti_plus_1_low)], r2=tx_zero_coupon.iloc[int(Ti_plus_1_up)], t=Ti_plus_1)
                    B_t_Ti_plus_1 = np.exp(-(r_Ti_plus_1/100) * Ti_plus_1)
            # Calcul du taux forward FRA(t, T_{i-1}, T_i)
            FRA_t = ( B_t_Ti/B_t_Ti_plus_1 - 1) / delta

            # Mise à jour des sommes
            numérateur += delta * B_t_Ti * FRA_t
            dénominateur += delta * B_t_Ti

        # Calcul du taux fixe K
        K = numérateur / dénominateur
        return K
    

    def swap_rate_flexible(self, tx_zero_coupon,discount_factors, delta,maturities):
        """
        Calcule le taux de swap S(t, T1, Tn).

        Paramètres :
        - tx_zero_coupon : DataFrame contenant les données (Maturité, Coefficient d'actualisation).
        - delta : Liste des périodes entre chaque Ti-1 et Ti.

        Retour :
        - S : Le taux de swap.
        """
    # Extraire les coefficients d'actualisation (B(t, T)) depuis la table
        #discount_factors = tx_zero_coupon["Coef"]

        # Initialisation des sommes pour le numérateur et le dénominateur
        numérateur = 0
        dénominateur = 0
        n = int(maturities/delta)
        B_t_T1 = 0
        B_t_Tn = 0
        for i in range(1, n+1):
            # Prix de l'obligation zéro-coupon pour T_i et T_{i-1}
            Ti = i*delta
            Ti_plus_1 = (i+1)*delta
            if Ti <=1:
                B_t_Ti = discount_factors.iloc[0]
            if Ti >1:
                if Ti % 1 == 0:
                    B_t_Ti = discount_factors.iloc[int(Ti)]
                else:
                    Ti_low, Ti_up = self.encadrer_reel(Ti)
                    r_Ti =  self.interpolation_loglin(t1=Ti_low, t2=Ti_up , r1=tx_zero_coupon.iloc[int(Ti_low)], r2=tx_zero_coupon.iloc[int(Ti_up)], t=Ti)
                    B_t_Ti = np.exp((-r_Ti/100) * Ti)

            if i == 1:
                B_t_T1 = B_t_Ti
            if i == n:
                B_t_Tn = B_t_Ti
            # Mise à jour des sommes
            numérateur += B_t_T1 - B_t_Tn
            dénominateur += delta * B_t_Ti
        S = numérateur / dénominateur

        return S
    

    def calculate_FRA(self, tx_zero_coupon,discount_factors, t, T, delta):
        """
        Calcule le taux fixe (K) qui annule la valeur du swap payeur en utilisant les données de taux ZC.

        Paramètres :
        - tx_zero_coupon : DataFrame contenant les données (Maturité, Taux ZC, Coefficient d'actualisation).
        - delta : Liste des périodes entre chaque Ti-1 et Ti.

        Retour :
        - K : Le taux fixe qui égalise la valeur du swap à 0.
        """
        # Extraire les coefficients d'actualisation (B(t, T)) depuis la table
        #discount_factors = tx_zero_coupon["Coef"]
        if T%1==0:
            B_t_T = discount_factors.iloc[T]
        else:
            T_low, T_up = self.encadrer_reel(T)
            r_T = self.interpolation_loglin(t1=T_low, t2=T_up , r1=tx_zero_coupon.iloc[int(T_low)], r2=tx_zero_coupon.iloc[int(T_up)], t=T)
            B_t_T = np.exp(-(r_T/100) * T)
        T_plus_delta = T + delta
        if T_plus_delta %1==0:
            B_t_T_plus_delta = discount_factors.iloc[T_plus_delta]
        else:
            T_low, T_up = self.encadrer_reel(T_plus_delta)
            r_T = self.interpolation_loglin(t1=T_low, t2=T_up , r1=tx_zero_coupon.iloc[int(T_low)], r2=tx_zero_coupon.iloc[int(T_up)], t=T_plus_delta)
            B_t_T_plus_delta = np.exp(-(r_T/100) * T_plus_delta)
        FRA_t = ( B_t_T/B_t_T_plus_delta - 1) / delta
        return FRA_t