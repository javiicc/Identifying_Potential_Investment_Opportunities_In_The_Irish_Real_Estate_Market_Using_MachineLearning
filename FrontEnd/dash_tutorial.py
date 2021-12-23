# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
       #         meta_tags=[{'name': 'viewport',
        #                    'content': 'width=device-width, initial-scale=1.0'}]
                )

df = pd.read_csv('data_predicted.csv', sep=',')


'''
# Generate a table
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])
'''

# The scatter mapbox
fig = px.scatter_mapbox(df,
                        lat="latitude",
                        lon="longitude",
                        color="residual",
                        size="price",
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        size_max=15,
                        zoom=10,
                        mapbox_style='carto-positron',
                 #       width=1600,
                        height=1300,
                        #title='Investment Opportunities',
                       )

markdown_text = '''
    # Ireland's Real Estate Market Opportunities
    *Test for my Data Science thesis*
    '''

# Layout section: Boostrap
# ------------------------------------------------------------------

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H1("Ireland's Real Estate Market Opportunities",
                        className='text-center text-primary, mb-4'),
                width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H6("Welcome to Javier Castaño thesis",
                        className='text-center'),
                width=12)
    ]),


    dbc.Row([

        dbc.Col([
            dcc.Graph(id='map', figure=fig),
        ],
            width={'size': 8},
            style={"height": "100%"},
            #xs=12, sm=12, md=12, lg=8, xl=8
        ),

        dbc.Col([
            dcc.Graph(id='example', figure={}),
        ],
            width={'size': 4},
            style={"height": "100%"},
            #xs=12, sm=12, md=12, lg=4, xl=4
        )

    ], class_name='h-100'), # Horizontal:start,center,end,between,around

],
    fluid=True,
    style={"height": "100vh"},

)


'''
    #html.Label( ['City:'], style={'font-weight': 'bold'}),  #'display': 'inline-block'
    dcc.Dropdown(
        id='bedroom-dropdown',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC',
        clearable=True,
        optionHeight=35,
        style={"width": "30%",
               'height': '100%',
               #'display': 'inline-block',
               'font-weight': 'bold',
               "margin-left": "45%",
               #"margin-right": "50%",
               #"margin": "auto"
               }
    ),
    # Graph component
    html.Div([
        dcc.Graph(
        id='example-graph',
        figure=fig,
        config={
            'autosizable': True,
            #'displayModeBar':,
            #'displaylogo':True,
        },
        style={'width': '49%'},
        ) 
    ]
    ),
    # Table
    html.Div([
        generate_table(df)
    ]),


])
    
@app.callback(
    Output('dd-output-container', 'children'),
    Input('bedroom-dropdown', 'value')
)
def update_output(value):
    return 'You have selected "{}"'.format(value)
'''





if __name__ == '__main__':
    app.run_server(debug=True, port=3000)