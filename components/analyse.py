from datetime import datetime, timedelta

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

class Analyse:
    def __init__(self):
        self.button_frq = html.Div(
                [
                    dbc.RadioItems(
                        id="radio-type",
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Day", "value": 'day'},
                            {"label": "Week", "value": 'week'},
                            {"label": "Month", "value": 'month'},
                        ],
                        value='day',
                    )
                ],
                className="radio-group",
            )
        
        self.color_name = ["primary", "secondary", "success", "warning", "danger", "info", "dark"]
        
        self.tab =  dcc.Tabs([
                dcc.Tab(label='Asset Pricing', children=[
                        dbc.Row(
                                [
                                    dbc.Col([dcc.Graph(id='ticker-pricing-graph')], width=9),
                                    dbc.Col([html.Br(), 
                                             
                                        dbc.Row([
                                            dbc.Col([
                                                html.H6("Choose the Ticker :", style={"color": "#2c3e50", "fontWeight": "normal" }) ,
                                            ], width=6),
                                            dbc.Col([
                                               dbc.Switch(
                                                        id="standalone-switch",
                                                        value=True,
                                                    ),
                                                ], width=3, className="d-flex justify-content-end" ),
                                            dbc.Col(id="standalone-value", width=3, className="d-flex justify-content-start")
                                        ]),
                                    dbc.Input(id="ticker-symbole",debounce=True, type='text', placeholder="Valid input...", 
                                              valid=True, className="mb-3"),
                                    dbc.Input(id="risk-free",debounce=True, type='number', placeholder="Valid input...", 
                                              valid=True, className="mb-3"),
                                    dbc.Button("Compute IV", id="open-volatility", n_clicks=0, className="w-100"),
                                    dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Implied Volatility")),
                                                dbc.ModalBody(dcc.Graph(id="volatility-graph")),
                                            ],
                                            id="modal-xl",
                                            size="xl",
                                            is_open=False,
                                        ),
                                    html.Br(),html.Br(),
                                    dbc.Row([
                                            dbc.Col([
                                                html.Label("Strike (K):", style={"fontWeight": "bold"}),
                                                dbc.Input(id="input-K", type="number", placeholder="Enter strike (K)...", className="mb-3")
                                            ], width=6),
                                            dbc.Col([
                                                html.Label("Maturity (T):", style={"fontWeight": "bold"}),
                                                dbc.Input(id="input-T", debounce=True, type="number", placeholder="Enter maturity (T)...", className="mb-3")
                                            ], width=6)
                                        ]),
                                    dbc.Row(
                                            dbc.Col(dbc.Card(
                                                dbc.CardBody(
                                                        [ 
                                                        dbc.Row([
                                                            html.H6("Interpolate options"),
                                                            dbc.Col(html.Div(id="output-div"))]),
                                                        ]
                                                    )
                                            ), width=12)
                                        )
                                    ], width=3),
                                    
                                    
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
                                [   dbc.Col([
                                    html.Br(),
                                    dbc.Card(
                                        dbc.CardBody(
                                                [ 
                                                html.H5(
                                                id="symbole-portofio"
                                            )
                                                ]
                                            )
                                    ),
                                    dbc.Row([
                                            dbc.Col([
                                            dcc.Graph(id='portfolio-graph')
                                         ], width=9),
                                        dbc.Col([
                                            html.Br(),
                                            dash_table.DataTable(id="data-table", filter_action="native", filter_options={"placeholder_text": "Filter..."}, page_size=10)
                                            ], width=3)
                                    ]),
                                    

                                ], width=9),
                                    dbc.Col([
                                    html.Br(),
                                    dbc.Card(
                                        dbc.CardBody(
                                                [ 
                                                html.Div(
                                                        [
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    html.Div(
                                                                    dcc.DatePickerSingle(
                                                                        id="date-picker",
                                                                        display_format="DD/MM/YYYY",
                                                                        placeholder="Select a date",
                                                                        date=date(2010, 1, 1),
                                                                        style={"width": "100%"},
                                                                    ),
                                                                    style={"width": "100%",  "marginBottom": "15px"}, 
                                                                ),
                                                                     ], width=6),
                                                                dbc.Col([
                                                                    dbc.Button("Correlation", id="open-correlation", n_clicks=0, className="w-100"),
                                                                    ], width=6)
                                                            ]),
                                                            dbc.Input(placeholder="Add Ticker...", id = 'add-ticker-management', valid=True, className="mb-3", debounce=True, type="text"),
                                                            dcc.Dropdown(
                                                            id="remove-ticker-dropdown",
                                                            placeholder="Remove Ticker...",
                                                            clearable=True,
                                                        ),
                                                        ]
                                                    )
                                                ]
                                            )
                                    ),
                                    html.Br(),
                                    dbc.Card(
                                    dbc.CardBody(
                                                [ 
                                                self.button_frq
                                                ]
                                            )
                                    ),
                                    html.Br(),
                                    dbc.Row([
                                            dbc.Col([
                                            html.Label("Min Weight:", style={"fontWeight": "bold"}),
                                            dbc.Input(id="input-weight-inf", type="number", placeholder="Enter Weight - ...", className="mb-3")
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Max Weight:", style={"fontWeight": "bold"}),
                                            dbc.Input(id="input-weight-sup", debounce=True, type="number", placeholder="Enter Weight + ...", className="mb-3")
                                        ], width=6)
                                    ]),
                                    dbc.Button("Efficient Frontier", color = "success", id="run-frontier", n_clicks=0, className="w-100"),
                                    
                                    ], width=3),
                                ])
                    
                        ]),
                ])
        
    def add_ticker(self, symbole_list, symbole):
        if symbole:
            return symbole_list.append(dbc.Badge(symbole, color="primary", className="border me-1"))
        else:
            return symbole_list

    def render(self):
        row = html.Div(
                [
                   self.tab,
                   dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Error")),
                            dbc.ModalBody("The ticker symbol entered is not recognized."),
                            dbc.ModalFooter(dbc.Button("OK", id="close-error-popup", className="ms-auto", n_clicks=0)),
                        ],
                        id="error-popup",
                        is_open=False,
                    ),
                ]
            )
        return row
