# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

app = dash.Dash(__name__)

df = pd.read_csv('data_example.csv', sep=',')

# Generates a table
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
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=15,
                        zoom=10,
                        mapbox_style='carto-positron',
                        width=1600, 
                        height=1000,
                        title='Investment Opportunities',
                       )





markdown_text = '''
    
    # Ireland's Real Estate Market Opportunities
    *Test for my Data Science thesis*
    
    '''


app.layout = html.Div([
    
    html.Div([
        dcc.Markdown(markdown_text), 
    ]), 
    
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
    
    html.Div([
        generate_table(df)
    ]), 


])
    
    

    
    #generate_table(df),





if __name__ == '__main__':
    app.run_server(debug=True)