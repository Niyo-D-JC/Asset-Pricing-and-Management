from datetime import datetime, timedelta

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px

class Analyse:
    def __init__(self):

        self.tab =  dcc.Tabs([
                dcc.Tab(label='Asset Pricing', children=[
                        dbc.Row(
                                [
                                    dbc.Col([dcc.Graph(id='ticker-pricing-graph')], width=9),
                                    dbc.Col([html.Br(), html.H6("Choose the Ticker :", style={"color": "#2c3e50", "fontWeight": "normal" }) ,
                                    dbc.Input(id="ticker-symbole",debounce=True, type='text', placeholder="Valid input...", 
                                              valid=True, className="mb-3")], width=3),
                                ])
                    
                        ]),
                dcc.Tab(label='Zero Coupon Rate', children=[
                        dbc.Row(
                                [

                                ])
                    
                        ]),
                dcc.Tab(label='Tracking Error', children=[
                        dbc.Row(
                                [

                                ])
                    
                        ]),
                dcc.Tab(label='Portfolio Management', children=[
                        dbc.Row(
                                [

                                ])
                    
                        ]),
                ])
        


    def render(self):
        row = html.Div(
                [
                   self.tab,
                   dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Error")),
                            dbc.ModalBody("The ticker symbol entered is not recognized."),
                            dbc.ModalFooter(dbc.Button("Close", id="close-error-popup", className="ms-auto", n_clicks=0)),
                        ],
                        id="error-popup",
                        is_open=False,
                    ),
                ]
            )
        return row
