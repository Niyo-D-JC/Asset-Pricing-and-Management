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

import plotly.express as px
import pandas as pd
from collections import Counter

# Initialisation du chemin permettant le lancement de l'application
# Définition du chemin de base pour l'application Dash en utilisant une variable d'environnement pour l'utilisateur

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

path = f"/"
app = Dash(__name__, requests_pathname_prefix=path, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], suppress_callback_exceptions = True)

app.index_string = INDEX_CONFIG

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
    Output("ticker-symbole", "value"),
    Input('load-data-button', 'n_clicks') 
)
def initialize_days(_):
    return "AAPL"


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

if __name__ == '__main__':
    app.run(debug=True)
    
