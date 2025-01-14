from dash import Dash, html, Input, Output, callback, dcc, State, dash_table, ctx
import os
from datetime import datetime, timedelta
import dash

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np

from components.menu import *
from components.analyse import Analyse
from components.asset_pricing import Pricing
from components.asset_management import Management

import plotly.express as px
import pandas as pd
from collections import Counter

# Initialisation du chemin permettant le lancement de l'application
# Définition du chemin de base pour l'application Dash en utilisant une variable d'environnement pour l'utilisateur

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

path = f"/"
app = Dash(__name__, requests_pathname_prefix=path, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], suppress_callback_exceptions = True)

app.index_string = INDEX_CONFIG

pricing = Pricing()

# Initialisation des différentes sections de l'application via des objets personnalisés
analyse = Analyse()      

CONTENT_STYLE = {
        "margin-left": "5.7rem",
        "margin-right": "5.7rem",
        "padding": "2rem 1rem",
    }

sidebar = Menu(path).render()
content =  dbc.Spinner(html.Div(id="page-content", style=CONTENT_STYLE), spinner_style={"width": "3rem", "height": "3rem"})

app.layout = html.Div(
    [
        dcc.Location(id="url"), # Permet de gérer les URLs et la navigation au sein de l'application
        sidebar, # Ajout de la barre latérale
        content, # Contenu principal avec un Spinner
        html.Button(id='load-data-button', style={"display": "none"}), # Bouton caché pour déclencher le chargement des données
        dcc.Store(id='selected-item', data='', storage_type='session'),  # Stockage temporaire de données sélectionnées en session
        html.Div(id="hidden-div", style={"display": "none"}), # Division cachée pour stocker d'autres informations ou déclencher des callbacks
    ])


options_dict = {
    True: "Call",  # Quand le switch est activé, "Call"
    False: "Put"   # Quand le switch est désactivé, "Put"
}

options_dict_value = {
    True: "call",  # Quand le switch est activé, "Call"
    False: "put"   # Quand le switch est désactivé, "Put"
}

# 
symbole_list = [
    "AAPL",  # Apple (Technologie)
    "EEM",   # iShares MSCI Emerging Markets ETF (Pays émergents)
    "TLT",   # US Treasury Bonds (Obligations souveraines)
    "HYG",   # High Yield Corporate Bonds
    "GLD",   # Gold (Or)
    "LDOS",  # Leidos Holdings Inc
    "LLY",   # Eli Lilly & Co
    "AMD",   # AMD
    "USO",   # United States Oil Fund (Pétrole)
    "VNQ",   # Vanguard Real Estate ETF (Immobilier)
    "BTC-USD", # Bitcoin (Cryptomonnaie)
]

symbole_list = list(set(symbole_list))
# Callback pour mettre à jour le contenu de la page en fonction du chemin d'URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # Affiche le rendu correspondant à l'URL, sinon retourne l'analyse par défaut
    if pathname == f"{path}":
        return analyse.render() 
    else:
        return analyse.render()             # Page par défaut (analyse) si le chemin n'est pas reconnu



############################ ANALYSE #################################

@app.callback(
    [Output("ticker-symbole", "value"), Output("risk-free", "value"), Output("symbole-portofio", "children"),
     Output('remove-ticker-dropdown', 'options'),],
    Input('load-data-button', 'n_clicks'),
)
def initialize_elements(_):
    bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
    return "AAPL", 0.03, bdg, symbole_list

@app.callback(
    Output("standalone-value", "children"),
    Input("standalone-switch", "value")
)
def update_option(value):

    return html.Div(f"{options_dict[value]}", style={"fontWeight": "bold", "color": "red"})

@app.callback(
    [Output('ticker-pricing-graph', 'figure'),
     Output('error-popup', 'is_open')],
    [Input('ticker-symbole', 'value'),
     Input('close-error-popup', 'n_clicks')],
)
def update_graph(ticker_, close_error_clicks):
    # Détection si la fermeture du popup est cliquée
    if ctx.triggered_id == "close-error-popup":
        return dash.no_update, False
    # Tentative de récupération des données pour le ticker
    try:
        dta_ = yf.download(ticker_, start='2010-01-01')
        print(dta_)
        if ticker_ is None or dta_.shape[0]<1:
            raise ValueError("Invalid ticker")

        dta_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        pricing.get_data(ticker_)
        pricing.price = float(dta_['Close'].values[-1]) 
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=dta_.index, y=dta_['Close'], mode='lines', name='Close'))
        
        return fig, False
    
    except Exception:
        # Retourne un graphique par défaut (ligne droite) et affiche le popup
        default_data = pd.DataFrame({
            'Date': pd.date_range(start='2010-01-01', periods=10),
            'Close': [10] * 10
        }).set_index('Date')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=default_data.index, y=default_data['Close'], mode='lines', name='Close'))
        return fig, True



@app.callback(
    [Output("modal-xl", "is_open"), Output("volatility-graph", "figure")],
    [Input("open-volatility", "n_clicks"), Input("risk-free", "value")],
    [State("modal-xl", "is_open")]
)
def handle_button_click(n_clicks, risk, is_open):
    if n_clicks > 0:
        # Vérifier si la colonne "Volatilité" existe, sinon la calculer
        if "IV" not in pricing.data.columns:
            pricing.compute_iv(risk)
        pricing.data = pricing.data.dropna(subset=['T', 'K', 'IV'])
        pivot = pricing.data.pivot(index='T', columns='K', values='IV')
        T = pivot.index.values  # Maturités (axe y)
        K = pivot.columns.values  # Strikes (axe x)
        IV = pivot.values  # Volatilités implicites (axe z)

        # Créer un graphique 3D avec Plotly
        T_mesh, K_mesh = np.meshgrid(T, K, indexing='ij')
        fig = go.Figure(
            data=[
                go.Surface(
                    z=IV, x=K_mesh, y=T_mesh, colorscale='Viridis'
                )
            ]
        )
        fig.update_layout(
            title="Implied Volatility Surface",
            scene=dict(
                xaxis_title="Strike Price (K)",
                yaxis_title="Maturity (T in years)",
                zaxis_title="Implied Volatility (IV)",
            ),
            height=600,
            width=800
        )

        return True, fig

    return is_open, dash.no_update



@app.callback(
    Output("output-div", "children"),
    State("risk-free", "value"),
    State("standalone-switch", "value"),
    State("input-K", "value"),
    Input("input-T", "value"),
    prevent_initial_call=True
)
def compute_iv(r, option_type, K, T):
    if K is None or T is None:
        return dbc.Alert("Please provide valid inputs for K and T.", color="danger")
    if "IV" not in pricing.data.columns:
            if r == None:
                r = 0.03
            pricing.compute_iv(r)

    S0 = pricing.price
    price, volatility= pricing.compute_price(K, T, S0, r=r, option_type=options_dict_value[option_type])

    delta, gamma, vega, theta = pricing.compute_greeks(K, T, S0, volatility, r=r, option_type=options_dict_value[option_type]) 

    # Mise en page en deux colonnes
    return dbc.Row([
        dbc.Col(html.Ul([
            html.Li(f"Price: {price:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Volatility: {volatility:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Delta: {delta:.2f}", style={"marginBottom": "10px"}),
        ]), width=6),

        dbc.Col(html.Ul([
            html.Li(f"Gamma: {gamma:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Vega: {vega:.2f}", style={"marginBottom": "10px"}),
            html.Li(f"Theta: {theta:.2f}", style={"marginBottom": "10px"})
        ]), width=6)
    ],)



@app.callback(
    [Output("symbole-portofio", "children", allow_duplicate=True),
     Output('error-popup', 'is_open', allow_duplicate=True), Output('remove-ticker-dropdown', 'options', allow_duplicate=True)],
    [Input('add-ticker-management', 'value'),
     Input('close-error-popup', 'n_clicks')],
     prevent_initial_call=True,
)
def add_symbole(ticker, close_error_clicks):
    global symbole_list
    # Détection si la fermeture du popup est cliquée
    if ctx.triggered_id == "close-error-popup":
        return dash.no_update, False, symbole_list
    
    # Tentative de récupération des données pour le ticker
    try:
        dta = yf.download(ticker, start='2010-01-01')
        if ticker is None or dta.shape[0]<1:
            raise ValueError("Invalid ticker")
        
        symbole_list.append(ticker)
        symbole_list = list(set(symbole_list))
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    
    except Exception:
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, True, symbole_list
    

@app.callback(
    [Output("symbole-portofio", "children", allow_duplicate=True),
     Output('error-popup', 'is_open', allow_duplicate=True), Output('remove-ticker-dropdown', 'options', allow_duplicate=True)],
    [Input('remove-ticker-dropdown', 'value')],
     prevent_initial_call=True,
)
def remove_symbole(ticker):
 
    global symbole_list

    if ticker is not None:
        symbole_list.remove(ticker)
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    else :
        bdg = [dbc.Badge(s, color=analyse.color_name[np.random.randint(0, 7)], className="border me-1") for s in symbole_list]
        return bdg, False, symbole_list
    

@app.callback(
    [Output('portfolio-graph', 'figure'), Output('data-table', 'data'),
     Output('data-table', 'columns')],
    Input('load-data-button', 'n_clicks')
)
def update_graph_portfolio(_):
    
    global symbole_list

    management = Management(symbole_list)
    management.get_parameters(freq = "day")

    rf = 0.03
    # Rendements cibles
    mu_targets = np.linspace(0.01, 0.2, 100)
    sml_volatilities = []
    sml_weights = []

    # Calcul de la frontière efficiente
    for mu_target in mu_targets:
        vol, weights = management.efficient_portfolio(mu_target, range_= (None, None))
        if vol is not None and weights is not None:
            sml_volatilities.append(vol)
            sml_weights.append(weights)

    # Identifier le portefeuille tangent (maximisation du ratio de Sharpe)
    sharpe_ratios = (np.array(mu_targets) - rf) / np.array(sml_volatilities)
    market_index = np.argmax(sharpe_ratios)
    market_volatility = sml_volatilities[market_index]
    market_return = mu_targets[market_index]
    market_weights = sml_weights[market_index]

    # Pente de la CML
    cml_slope = (market_return - rf) / market_volatility

    # Étendre la CML
    extended_volatilities = np.linspace(0, max(sml_volatilities) * 1.1, 200)
    cml_y = rf + cml_slope * extended_volatilities

    fig = go.Figure()

    # Ajouter la frontière efficiente
    fig.add_trace(go.Scatter(
        x=sml_volatilities, 
        y=mu_targets,
        mode='lines',
        name='Efficient border',
        line=dict(color='blue')
    ))

    # Ajouter la CML étendue
    fig.add_trace(go.Scatter(
        x=extended_volatilities, 
        y=cml_y,
        mode='lines',
        name='CML (Capital Market Line)',
        line=dict(color='red', dash='dash')
    ))

    # Marquer le portefeuille du marché
    fig.add_trace(go.Scatter(
        x=[market_volatility], 
        y=[market_return],
        mode='markers',
        name='Market Portfolio',
        marker=dict(color='green', size=10)
    ))

    # Marquer le taux sans risque
    fig.add_trace(go.Scatter(
        x=[0], 
        y=[rf],
        mode='markers',
        name='Risk-free rate',
        marker=dict(color='black', size=10)
    ))

    # Configurer le graphique
    fig.update_layout(
        title='Extended CML with Market Portfolio',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Yield-Return',
        legend=dict(x=0.02, y=0.98),
        template='plotly_white'
    )

    df_market_weights = pd.DataFrame({
        'Symbol': management.assets,
        'Weight': np.round(market_weights, 5)
    })
    
    table_data = df_market_weights.to_dict('records')
    table_columns = [{'name': col, 'id': col} for col in df_market_weights.columns]

    return fig, table_data, table_columns


if __name__ == '__main__':
    app.run(debug=True)
    
