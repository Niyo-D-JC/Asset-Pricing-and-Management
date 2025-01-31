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
                                    dbc.Col([dcc.Graph(id='ticker-pricing-graph'), html.Div(id="output-greeks")], width=9),
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
                                    dbc.Row([
                                            dbc.Col([
                                                dbc.Button("Compute IV", id="open-volatility", n_clicks=0, className="w-100"),
                                                ], width=6),
                                            dbc.Col([
                                                dbc.Button("Show Interpolation", id="open-interpol", disabled=True , color="secondary", n_clicks=0, className="w-100"),
                                                ], width=6)
                                        ]),
                                    dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Implied Volatility")),
                                                dbc.ModalBody(dcc.Graph(id="volatility-graph")),
                                            ],
                                            id="modal-xl",
                                            size="xl",
                                            is_open=False,
                                        ),
                                    dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Implied Volatility")),
                                                dbc.ModalBody(dcc.Graph(id="interpol-graph")),
                                            ],
                                            id="modal-xl-interpol",
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
                dcc.Tab(label='Vanilla Rate Pricing and Exotic Options', children=[
                        dbc.Row(
                                [
                                    dbc.Col(
                                [   html.Br(),
                                    dcc.Tabs(
                                            [
                                                dcc.Tab(
                                                    label='Vanilla Rate',
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col([
                                                                    html.Br(),
                                                                    html.H5("Zero Coupon Rate Data - 31st December 2024", style={"textAlign": "center"}),
                                                                        dash_table.DataTable(
                                                                            id="zero-coupons-table",
                                                                            style_table={"overflowX": "auto"},
                                                                            style_cell={"textAlign": "center", "fontFamily": "Arial", "fontSize": "14px"},
                                                                            style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
                                                                            filter_action="native", filter_options={"placeholder_text": "Filter..."},
                                                                            page_size=10,
                                                                        ),
                                                                        ],
                                                                    width=3,
                                                                ),
                                                                # Graph for Tracking Error
                                                                dbc.Col(
                                                                    [  
                                                                        dcc.Graph(id="interpolate-graph"),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [  
                                                                    html.Br(),
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [   
                                                                                html.Label("Pricing method/Rate", style={"fontWeight": "normal","marginBottom": "5px"}),
                                                                                dcc.Dropdown(
                                                                                    id="princing-method",
                                                                                    options=[
                                                                                        {"label": "Flexible Bond", "value": "flex-bond"},
                                                                                        {"label": "Swap Rate Flexible", "value": "swap-rate"},
                                                                                        {"label": "FRA Calculate", "value": "fra"},
                                                                                    ],
                                                                                    value="flex-bond",
                                                                                    placeholder="Select frequency",
                                                                                    style={"fontWeight": "normal"}
                                                                                ),
                                                                                html.Br(),
                                                                                dbc.Col(html.Div(id="input-princing-div")),
                                                                                dbc.Button(
                                                                                    "Run Pricing/Rate",
                                                                                    id="run-pricing",
                                                                                    color="primary",
                                                                                    className="w-100",
                                                                                    n_clicks=1,
                                                                                ),
                                                                                html.Br(), html.Br(), 
                                                                                dbc.Col(html.Div(id="output-princing-div"))
                                                                            ]
                                                                        ),
                                                                        className="mb-3",
                                                                    ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f9f9f9",
                                                        "color": "#28A745",
                                                    },
                                                    selected_style={
                                                        "backgroundColor": "#28A745",
                                                        "color": "#ffffff",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Tab(
                                                    label='Barrier Options',
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col([
                                                                    html.Br(),
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [   html.H5("Simulate trajectories", style={"marginBottom": "5px"}),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("S0:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="price_s0-bar", value= 100, type="number", placeholder="Initial stock price", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Maturity:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="maturity-bar", value= 1,debounce=True, type="number", placeholder="Maturity", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Risk free:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="risk-free-bar", value= 0.03,type="number", placeholder="Risk-free rate", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Volatility:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="volatility-bar", value= 0.2,debounce=True, type="number", placeholder="Volatility", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Strike:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="strike-bar", value= 90, type="number", placeholder="Strike", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Barrier:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="barrier-bar", value= 110, type="number", placeholder="Barrier", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Barrier type:", style={"fontWeight": "normal"}),
                                                                                            dcc.Dropdown(
                                                                                                id="barrier-type",
                                                                                                options=[
                                                                                                    {"label": "Up-and-Out", "value": "up-and-out"},
                                                                                                    {"label": "Down-and-Out", "value": "down-and-out"},
                                                                                                ],
                                                                                                value="up-and-out",
                                                                                                placeholder="Select frequency",
                                                                                                style={"fontWeight": "normal"}
                                                                                            ),
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Option type:", style={"fontWeight": "normal"}),
                                                                                            dcc.Dropdown(
                                                                                                id="option-type",
                                                                                                options=[
                                                                                                    {"label": "Call Option", "value": "call"},
                                                                                                    {"label": "Put Option", "value": "put"},
                                                                                                ],
                                                                                                value="call",
                                                                                                placeholder="Select frequency",
                                                                                                style={"fontWeight": "normal"}
                                                                                            ),
                                                                                        ], width=6),
                                                                                        
                                                                                    ]),
                                                                                html.Br(),
                                                                                dbc.Button(
                                                                                    "Run Simulation",
                                                                                    id="run-barrier",
                                                                                    color="primary",
                                                                                    className="w-100",
                                                                                    n_clicks=1,
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        className="mb-3",
                                                                    ),
                                                                        ],
                                                                    width=3,
                                                                ),
                                                                # Graph for Tracking Error
                                                                dbc.Col(
                                                                    [  
                                                                        dcc.Graph(id="simulate-barrier-graph")
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [  
                                                                    html.Br(),
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [   
                                                                               html.H5("Barrier Option Price/Greeks:", style={"marginBottom": "5px"}),
                                                                               html.Br(),
                                                                               dbc.Col(html.Div(id="output-barrier-div"))

                                                                            ]
                                                                        ),
                                                                        className="mb-3",
                                                                    ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f9f9f9",
                                                        "color": "#28A745",
                                                    },
                                                    selected_style={
                                                        "backgroundColor": "#28A745",
                                                        "color": "#ffffff",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Tab(
                                                    label='Asian Options',
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                
                                                                dbc.Col([
                                                                    html.Br(),
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [   html.H5("Simulate trajectories", style={"marginBottom": "5px"}),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("S0:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="price_s0-asian", value= 100, type="number", placeholder="Initial stock price", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Maturity:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="maturity-asian", value= 1, debounce=True, type="number", placeholder="Maturity", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Risk free:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="risk-free-asian", value= 0.03, type="number", placeholder="Risk-free rate", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Volatility:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="volatility-asian", value= 0.2, debounce=True, type="number", placeholder="Volatility", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Strike:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="strike-asian", value= 90, type="number", placeholder="Strike", className="mb-3")
                                                                                        ], width=6),
                                                                                        dbc.Col([
                                                                                            html.Label("Time window:", style={"fontWeight": "normal"}),
                                                                                            dbc.Input(id="delta-asian", value= 0.25, type="number", placeholder="Averaging window", className="mb-3")
                                                                                        ], width=6)
                                                                                    ]),
                                                                                dbc.Row([
                                                                                        dbc.Col([
                                                                                            html.Label("Option type:", style={"fontWeight": "normal"}),
                                                                                            dcc.Dropdown(
                                                                                                id="option-type-asian",
                                                                                                options=[
                                                                                                    {"label": "Call Option", "value": "call"},
                                                                                                    {"label": "Put Option", "value": "put"},
                                                                                                ],
                                                                                                value="call",
                                                                                                placeholder="Select frequency",
                                                                                                style={"fontWeight": "normal"}
                                                                                            ),
                                                                                        ], width=12),
                                                                                        
                                                                                    ]),
                                                                                html.Br(),
                                                                                
                                                                                dbc.Button(
                                                                                    "Run Simulation",
                                                                                    id="run-asian",
                                                                                    color="primary",
                                                                                    className="w-100",
                                                                                    n_clicks=1,
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        className="mb-3",
                                                                    ),
                                                                        ],
                                                                    width=3,
                                                                ),
                                                                # Graph for Tracking Error
                                                                dbc.Col(
                                                                    [  
                                                                        dcc.Graph(id="simulate-asian-graph")
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [  
                                                                    html.Br(),
                                                                    dbc.Card(
                                                                        dbc.CardBody(
                                                                            [   
                                                                               html.H5("Asian Option Price/Greeks:", style={"marginBottom": "5px"}),
                                                                               html.Br(),
                                                                               dbc.Col(html.Div(id="output-asian-div"))

                                                                            ]
                                                                        ),
                                                                        className="mb-3",
                                                                    ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f9f9f9",
                                                        "color": "#28A745",
                                                    },
                                                    selected_style={
                                                        "backgroundColor": "#28A745",
                                                        "color": "#ffffff",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                
                                            ],
                                            style={
                                                "border": "1px solid #dcdcdc",  # Bordure autour des onglets
                                                "backgroundColor": "#f1f1f1",  # Couleur de fond des onglets
                                            },
                                        )

                                    ],
                                width=12,
                            ),
                                ])
                    
                                        ]),
                dcc.Tab(label='Tracking Error', children=[
                    dbc.Row(
                        [   
                            dbc.Col([
                                html.Br(),
                                dbc.Card(
                                    dbc.CardBody(
                                        [   dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Start: ", style={"fontWeight": "normal", "marginRight": "10px"}),
                                                    dcc.DatePickerSingle(
                                                        id="start-date-picker",
                                                        min_date_allowed=date(2010, 1, 1),
                                                        max_date_allowed=date.today(),
                                                        initial_visible_month=date(2020, 1, 1),
                                                        date=date(2020, 1, 1),
                                                        style={"fontWeight": "normal"}
                                                    ),
                                                ],
                                                width="auto"  # Ajuste la largeur au contenu
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("End: ", style={"fontWeight": "normal", "marginRight": "10px"}),
                                                    dcc.DatePickerSingle(
                                                        id="end-date-picker",
                                                        min_date_allowed=date(2010, 1, 1),
                                                        max_date_allowed=date.today(),
                                                        initial_visible_month=date.today(),
                                                        date=date.today(),
                                                        style={"fontWeight": "normal"}
                                                    ),
                                                ],
                                                width="auto",  # Ajuste la largeur au contenu
                                            ),
                                        ],
                                        justify="between"  # Place les colonnes aux extrémités
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                html.Label("Data Rebalancing", style={"fontWeight": "normal"}),
                                                dcc.Dropdown(
                                                    id="data-rebalancing",
                                                    options=[
                                                        {"label": "Yearly", "value": "Y"},
                                                        {"label": "Monthly", "value": "M"},
                                                    ],
                                                    value="Y",
                                                    placeholder="Select frequency",
                                                    style={"fontWeight": "normal"}
                                                ),
                                                ],
                                                width=6  # Ajuste la largeur au contenu
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Br(),
                                                    dbc.Input(id="max-assets", min = 20, max=40, debounce=True, type='number', placeholder="Max Assets : 20-40", 
                                              valid=True, className="mb-3"),
                                                ],
                                                width=6,  # Ajuste la largeur au contenu
                                            ),
                                        ],
                                        justify="between"  # Place les colonnes aux extrémités
                                    ),
        
                                            dbc.Row([
                                                    dbc.Col([
                                                        html.H6("Show sector weights:", style={"color": "#2c3e50", "fontWeight": "normal"})
                                                    ], width=6),
                                                    dbc.Col([
                                                        dbc.Switch(
                                                            id="sector-standalone-switch",
                                                            value=False,
                                                        )
                                                    ], width="auto", className="d-flex justify-content-end")
                                                ], className="justify-content-between"),
                                            html.Br(),
                                            dbc.Button(
                                                "Update",
                                                id="run-backtest",
                                                color="primary",
                                                className="w-100",
                                                n_clicks=1,
                                            ),
                                        ]
                                    ),
                                    className="mb-3",
                                ),
                                
                                html.H5("Optimized Portfolio Weights", style={"textAlign": "center"}),
                                    dash_table.DataTable(
                                        id="optimized-weights-table",
                                        columns=[
                                            {"name": "Ticker", "id": "Ticker"},
                                            {"name": "Sector", "id": "Sector"},
                                            {"name": "Symbol", "id": "Symbol"},
                                            {"name": "Weight (%)", "id": "Weight"},
                                        ],
                                        style_table={"overflowX": "auto"},
                                        style_cell={"textAlign": "center", "fontFamily": "Arial", "fontSize": "14px"},
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
                                        filter_action="native", filter_options={"placeholder_text": "Filter..."},
                                        page_size=10,
                                    ),
                                    ],
                                width=4,
                            ),
                            # Graph for Tracking Error
                            dbc.Col(
                                [   html.Br(),
                                    dcc.Tabs(
                                            [
                                                dcc.Tab(
                                                    label='Tracking Error',
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dcc.Graph(id="tracking-error-graph"),
                                                            ]
                                                        )
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f9f9f9",
                                                        "color": "#28A745",
                                                    },
                                                    selected_style={
                                                        "backgroundColor": "#28A745",
                                                        "color": "#ffffff",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Tab(
                                                    label='Backtest',
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dcc.Graph(id="annualized-returns-graph"),
                                                            ]
                                                        )
                                                    ],
                                                    style={
                                                        "backgroundColor": "#f9f9f9",
                                                        "color": "#28A745",
                                                    },
                                                    selected_style={
                                                        "backgroundColor": "#28A745",
                                                        "color": "#ffffff",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "border": "1px solid #dcdcdc",  # Bordure autour des onglets
                                                "backgroundColor": "#f1f1f1",  # Couleur de fond des onglets
                                            },
                                        )

                                    ],
                                width=8,
                            ),
                        ],
                        className="mb-3",
                    )
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
                                                                    dbc.Modal(
                                                                            [
                                                                                dbc.ModalHeader(dbc.ModalTitle("Correlation Matrix")),
                                                                                dbc.ModalBody(dcc.Graph(id="correlation-graph")),
                                                                            ],
                                                                            id="modal-xl-corr",
                                                                            size="xl",
                                                                            is_open=False,
                                                                        ),
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
                                            dbc.Input(id="input-weight-inf", type="number", value=-0.1, placeholder="Enter Weight - ...", className="mb-3")
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
