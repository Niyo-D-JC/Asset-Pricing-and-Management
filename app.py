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
    [Output("ticker-symbole", "value"), Output("risk-free", "value")],
    Input('load-data-button', 'n_clicks') 
)
def initialize_days(_):
    return "AAPL", 0.03


@app.callback(
    [Output('ticker-pricing-graph', 'figure'),
     Output('error-popup', 'is_open')],
    [Input('ticker-symbole', 'value'),
     Input('close-error-popup', 'n_clicks')],
)
def update_graph(ticker, close_error_clicks):
    # Détection si la fermeture du popup est cliquée
    if ctx.triggered_id == "close-error-popup":
        return dash.no_update, False
    
    # Tentative de récupération des données pour le ticker
    try:
        dta = yf.download(ticker, start='2010-01-01')
        if ticker is None or dta.shape[0]<1:
            raise ValueError("Invalid ticker")

        dta.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        pricing.get_data(ticker)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dta.index, y=dta['Close'], mode='lines', name='Close'))
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
        print(risk)
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
    State("input-K", "value"),
    Input("input-T", "value"),
    prevent_initial_call=True
)
def compute_iv(K, T):
    if K is None or T is None:
        return dbc.Alert("Please provide valid inputs for K and T.", color="danger")

    price = 100  # Exemple : prix simulé
    volatility = 0.2  # Exemple : volatilité simulée
    delta = 0.5  # Exemple : delta simulé
    gamma = 0.1  # Exemple : gamma simulé
    vega = 0.2  # Exemple : vega simulé
    theta = -0.01  # Exemple : theta simulé

    # Mise en page en deux colonnes
    return dbc.Row([
        dbc.Col(html.Ul([
            html.Li(f"Price: {price}", style={"marginBottom": "10px"}),
            html.Li(f"Volatility: {volatility}", style={"marginBottom": "10px"}),
            html.Li(f"Delta: {delta}", style={"marginBottom": "10px"}),
        ]), width=6),

        dbc.Col(html.Ul([
            html.Li(f"Gamma: {gamma}", style={"marginBottom": "10px"}),
            html.Li(f"Vega: {vega}", style={"marginBottom": "10px"}),
            html.Li(f"Theta: {theta}", style={"marginBottom": "10px"})
        ]), width=6)
    ],)


if __name__ == '__main__':
    app.run(debug=True)
    
