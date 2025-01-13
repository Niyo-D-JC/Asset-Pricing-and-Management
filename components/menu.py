from dash import html
import dash_bootstrap_components as dbc
from datetime import datetime


INDEX_CONFIG = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Adobe Price Prophet</title>
        <link rel="icon" type="image/png" href="https://static.cdnlogo.com/logos/a/90/adobe.png">  <!-- Référence à votre favicon -->
        {%metas%}
        {%css%}
    </head>
    <body>
        <!--[if IE]><script>
        alert("Dash v2.7+ does not support Internet Explorer. Please use a newer browser.");
        </script><![endif]-->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

class Menu:
    def __init__(self, path):
        self.path = path
        self.SIDEBAR_STYLE = {
                "position": "fixed",
                "top": 0,
                "left": 0,
                "bottom": 0,
                "width": "16rem",
                "padding": "2rem 1rem",
                "background-color": "#f8f9fa",
            }
    def get_current_year(self):
        return datetime.now().year
    
    def render(self):
        return dbc.NavbarSimple(
                    children=[
                        dbc.NavItem(dbc.NavLink("ENSAI 3A Project")),
                    ],
                    brand="Asset Pricing & Management",
                    color="primary",
                    dark=True,
                )
