# Run this app with `python app.py` and
# visit http://127.0.0.1:3000/ in your web browser.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX],
               # meta_tags=[{'name': 'viewport',
                #            'content': 'width=device-width, initial-scale=1.0'}]
                )

df = pd.read_csv('../investment-opportunities/data/07_model_output/data_for_frontend.csv',
                 sep=',')
df['predicted_price'] = df['predicted_price'].round(decimals=0)
df['residual'] = df['residual'].round(decimals=0)


# Layout section: Boostrap
# ------------------------------------------------------------------
app.layout = dbc.Container([

    # First row (height 15)
    dbc.Row([
        # Column (total width 12)
        # Title
        dbc.Col(html.H1("Ireland's Real Estate Market Opportunities",
                        className='pt-4 text-center text-primary text-light',
                        ),
                width=12,
                style={"backgroundColor": "#100508"},
                ),
        # Explanation bellow the title
        dcc.Markdown('''
        Welcome to Javier Castaño's final project from the Master in Data Science at
        KSchool. You can interact with the map through the controls bellow.''',
                     className="text-center",
                     style={"backgroundColor": "#100508"}
                     ),
    ],
        style={"height": "15vh"}
    ),

    # Second row (height 80)
    # Contains 3 columns
    dbc.Row([
        # First column (width 8 of 12)
        dbc.Col([
            # Map graph
            dcc.Graph(
                id='map',
                responsive=True,
                style={'height': '100%'}
            ),
        ],
            width={'size': 8},
            style={"backgroundColor": "#100508"}
        ),
        # Second column (width 2 of 12)
        dbc.Col([
            # Bar graph
            dcc.Graph(
                id='bar',
            ),
            # Markdown bellow bar graph -> price_range_slider title
            dcc.Markdown(
                '''
                **PRICE**
                ''',
                className="text-center my-5",
            ),
            # Range slider for price variable
            dcc.RangeSlider(
                id='price_range_slider',
                className='my-sm-5',
                min=int(df.actual_price.min()),
                max=int(df.actual_price.max()),
                value=[int(df.actual_price.min()), int(df.actual_price.max())],
                marks={
                    int(df.actual_price.min()): {
                        'label': f'{int(df.actual_price.min())}€',
                        'style': {'color': '#f50'}},
                    int((df.actual_price.max() + df.actual_price.min()) / 2): {
                        'label': f'{int((df.actual_price.max() + df.actual_price.min())/ 2)}€',
                        'style': {'color': '#f50'}},
                    int(df.actual_price.max()): {
                        'label': f'{int(df.actual_price.max())}€',
                        'style': {'color': '#f50'}}},
                tooltip={"placement": "top", "always_visible": False}
            ),
            # Markdown  -> residuals_range_slider title
            dcc.Markdown(
                '''
                **RESIDUALS**
                ''',
                className="text-center",
            ),
            # Range slider for residuals variable
            dcc.RangeSlider(
                id='residuals_range_slider',
                className='my-sm-5',
                min=int(df.residual.min()),
                max=int(df.residual.max()),
                value=[int(df.residual.min()), int(df.residual.max())],
                marks={
                    int(df.residual.min()): {
                        'label': f'{int(df.residual.min())}€',
                        'style': {'color': '#f50'}},
                    0: {
                        'label': f'0€',
                        'style': {'color': '#f50'}},
                    int(df.residual.max()): {
                        'label': f'{int(df.residual.max())}€',
                        'style': {'color': '#f50'}}},
                tooltip={"placement": "top", "always_visible": False}
            ),
        ],
            width={'size': 2},
            style={"backgroundColor": "#100508"}
        ),
        # Third column (width 2 of 12)
        dbc.Col([
            # Pie graph
            dcc.Graph(
                id='pie',
            ),
            # Markdown bellow pie graph -> floor_area_slider title
            dcc.Markdown(
                '''
                **FLOOR AREA**
                ''',
                className="text-center my-5",
            ),
            # Range slider for floor_area variable
            dcc.RangeSlider(
                id='floor_area_slider',
                className='my-sm-5',
                min=int(df.floor_area.min()),
                max=int(df.floor_area.max()),
                # step=None,
                value=[int(df.floor_area.min()), int(df.floor_area.max())],
                marks={
                    int(df.floor_area.min()): {
                        'label': f'{int(df.floor_area.min())}m²',
                        'style': {'color': '#f50'}},
                    int((df.floor_area.max() + df.floor_area.min()) / 2): {
                        'label': f'{int((df.floor_area.max() + df.floor_area.min()) / 2)}'
                                 f'm²',
                        'style': {'color': '#f50'}},
                    int(df.floor_area.max()): {
                        'label': f'{int(df.floor_area.max())}m²',
                        'style': {'color': '#f50'}}},
                tooltip={"placement": "top", "always_visible": False}
            ),
            # Markdown -> bedroom_range_slider title
            dcc.Markdown(
                '''
                **BEDROOMS**
                ''',
                className="text-center",
            ),
            # Range slider for bedroom variable
            dcc.RangeSlider(
                id='bedroom_range_slider',
                className='my-sm-3',
                marks={i: '{}'.format(i) for i in range(df.bedroom.min(),
                                                        df.bedroom.max())},
                min=df.bedroom.min(),
                max=df.bedroom.max(),
                value=[df.bedroom.min(), df.bedroom.max()]
            ),
            # Markdown -> bathroom_range_slider title
            dcc.Markdown(
                '''
                **BATHROOMS**
                ''',
                className="text-center",
            ),
            # Range slider for bathroom variable
            dcc.RangeSlider(
                id='bathroom_range_slider',
                className='my-sm-3',
                marks={i: '{}'.format(i) for i in range(df.bathroom.min(),
                                                        df.bathroom.max())},
                min=df.bathroom.min(),
                max=df.bathroom.max(),
                value=[df.bathroom.min(), df.bathroom.max()]
            ),
        ],
            width={'size': 2},
            style={"backgroundColor": "#100508"}
        )
    ],
        style={"height": "80vh"}
    ),
    # Third row (height 5)
    dbc.Row([
        # Empty column (width 8 of 12)
        dbc.Col(
                width=8,
                style={"backgroundColor": "#100508"}
                ),
        # Second column (width 4 of 12)
        dbc.Col(
            # Link to my LinkedIn ;)
            dcc.Link(
                'LinkedIn',
                href='https://www.linkedin.com/in/javier-casta%C3%B1o-candela-b89039208/',
                className="text-center",
                style={"backgroundColor": "#100508", 'color': '#77C6FB'},
                refresh=True,
            ),
            width=4,
            style={"backgroundColor": "#100508"}
                ),
    ],
        style={"height": "5vh"}
    ),

],
    fluid=True,
    style={"height": "100vh"}
)

# Callback section: Boostrap
# ------------------------------------------------------------------

#  By writing this decorator, we're telling Dash to call this function for us whenever
#  the value of the "input" component changes in order to update the children of
#  the "output" component on the page


@app.callback(
    Output(component_id='map', component_property='figure'),
    Output(component_id='pie', component_property='figure'),
    Output(component_id='bar', component_property='figure'),
    Input(component_id='price_range_slider', component_property='value'),
    Input(component_id='floor_area_slider', component_property='value'),
    Input(component_id='residuals_range_slider', component_property='value'),
    Input(component_id='bedroom_range_slider', component_property='value'),
    Input(component_id='bathroom_range_slider', component_property='value'),
)
def update_output_map(price_range, floor_area_range, residuals_range, bedroom_range,
                      bathroom_range):
    # Filter the DataFrame according to the inputs
    filtered_df = df[
        (df.actual_price >= price_range[0])
        & (df.actual_price <= price_range[1])
        & (df.floor_area >= floor_area_range[0])
        & (df.floor_area <= floor_area_range[1])
        & (df.residual >= residuals_range[0])
        & (df.residual <= residuals_range[1])
        & (df.bedroom >= bedroom_range[0])
        & (df.bedroom <= bedroom_range[1])
        & (df.bathroom >= bathroom_range[0])
        & (df.bathroom <= bathroom_range[1])
        ]
    # Map figure with the filtered DataFrame
    map_fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="res_percentage",
        size="actual_price",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=15,
        zoom=7.3,
        mapbox_style='carto-positron',
        range_color=[-1, 1],
        labels={'actual_price': 'PRICE',
                'predicted_price': 'PREDICTION',
                'residual': 'RESIDUAL'},
        hover_data={'actual_price': True,
                    'predicted_price': True,
                    'latitude': False,
                    'longitude': False,
                    'res_percentage': False,
                    'residual': True,
                    # 'url': True,
                    },
        hover_name='residual',
        opacity=1,
        template='plotly_dark',
    )
    map_fig.update(layout_coloraxis_showscale=False)
    # Pie figure with the filtered DataFrame
    pie_fig = px.pie(
        filtered_df,
        values='actual_price',
        names='type_house',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hover_data={'type_house': False,
                    'actual_price': False},
        template='plotly_dark',
    )
    # Mean price of the five places with higher mean prices according to the
    # filtered DataFrame
    mean_df = filtered_df.groupby('place')['actual_price'].mean()\
                         .sort_values(ascending=False).head()
    # Bar figure with DataFrame above
    bar_fig = px.bar(
        mean_df,
        x=mean_df.index,
        y=mean_df.values,
        labels={'place': '', 'y': ''},
        color_discrete_sequence=px.colors.sequential.RdBu,
        template='plotly_dark',
    )

    return map_fig, pie_fig, bar_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
