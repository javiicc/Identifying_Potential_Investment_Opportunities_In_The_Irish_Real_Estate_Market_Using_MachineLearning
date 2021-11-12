# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

df = pd.read_csv('data_example.csv', sep=',')

fig = px.scatter_mapbox(df,
                        lat="latitude",
                        lon="longitude",
                        color="dif_per",
                        size="price",
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=15,
                        zoom=10,
                        mapbox_style="carto-positron")
fig.show()

#app.layout = html.Div(children=[

 #   html.H1(children="Ireland's Real Estate Market"),

  #  html.Div(children='''
   #     Dash: A web application framework for your data.
    #'''),

#    dcc.Graph(
 #       id='example-graph',
  #      figure=fig
   # ),

#])

#fig.show()

if __name__ == '__main__':
    app.run_server(debug=True)