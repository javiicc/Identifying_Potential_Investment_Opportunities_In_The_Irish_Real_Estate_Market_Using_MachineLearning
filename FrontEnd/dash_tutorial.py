# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
markdown_text = '''
    # Ireland's Real Estate Market Opportunities
    *Test for my Data Science thesis*
    '''

df = pd.read_csv('data_predicted.csv', sep=',')


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


# The scatter mapbox
fig = px.scatter_mapbox(df,
                        lat="latitude",
                        lon="longitude",
                        color="dif_per",
                        size="price",
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        size_max=15,
                        zoom=10,
                        mapbox_style='carto-positron',
                        width=1600, 
                        height=1000,
                        #title='Investment Opportunities',
                       )


app.layout = html.Div([
    # Title and explanation
    html.Div([
        dcc.Markdown(markdown_text), 
    ]),

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






if __name__ == '__main__':
    app.run_server(debug=True)