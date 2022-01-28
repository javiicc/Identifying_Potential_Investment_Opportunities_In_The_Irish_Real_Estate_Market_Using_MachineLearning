# Run this app with `python app.py` and
# visit http://127.0.0.1:3000/ in your web browser.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import json
import re
import joblib
import os.path as path
import numpy as np

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
app.layout = dbc.Container(
    className="",
    children=[
        # First row (height 15)
        dbc.Row([
            # Column (total width 12)
            # Title
            dbc.Col(html.H1("Ireland's Real Estate Market Opportunities",
                            className='pt-4 text-center',  # text-primary   text-danger
                            style={'color': 'rgb(132,11,11)'}  #5D0808  rgb(117,12,12)
                            ),
                    width=12,
                    ),
            # Explanation bellow the title
            dcc.Markdown('''
            Welcome to Javier Castaño's final project from the Master in Data Science at
            KSchool. You can interact with the map through the controls bellow.
            ''',
                         className="text-center",
                         style={'color': 'rgb(216,173,173)'}
                         ),
        ],
            style={"height": "15vh"}
        ),

        # Second row (height 80)
        # Contains 3 columns
        dbc.Row(
            children=[
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
                ),
                # Second column (width 2 of 12)
                dbc.Col(
                    className="",
                    children=[
                        # Bar graph
                        dcc.Graph(
                            id='bar',
                        ),
                        # Markdown bellow bar graph -> price_range_slider title
                        dcc.Markdown(
                            '''
                            **PRICE**
                            ''',
                            className="text-center pt-5",
                            style={'color': 'rgb(103,214,140)'}
                        ),
                        # Range slider for price variable
                        dcc.RangeSlider(
                            id='price_range_slider',
                            className='my-3',
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
                            style={'color': 'rgb(103,214,140)'}
                        ),
                        # Range slider for residuals variable
                        dcc.RangeSlider(
                            id='residuals_range_slider',
                            className='my-3',
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
                        # Markdown bellow pie graph -> floor_area_slider title
                        dcc.Markdown(
                            '''
                            **FLOOR AREA**
                            ''',
                            className="text-center my-3",
                            style={'color': 'rgb(103,214,140)'}
                        ),
                        # Range slider for floor_area variable
                        dcc.RangeSlider(
                            id='floor_area_slider',
                            className='my-3',
                            min=int(df.floor_area.min()),
                            max=int(df.floor_area.max()),
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
                            style={'color': 'rgb(103,214,140)'}
                        ),
                        # Range slider for bedroom variable
                        dcc.RangeSlider(
                            id='bedroom_range_slider',
                            className='my-3',
                            marks={i: '{}'.format(i) for i in range(df.bedroom.min(),
                                                                    df.bedroom.max()+1)},
                            min=df.bedroom.min(),
                            max=df.bedroom.max(),
                            value=[df.bedroom.min(), df.bedroom.max()]
                        ),
                    ],
                    width={'size': 2},
                ),
                # Third column (width 2 of 12)
                dbc.Col([
                    dbc.Row(
                        className="my-2 py-2",  # border border-light
                        children=[
                            dcc.Markdown(
                                'Select to get the average prices in the bar chart',
                                className="text-center text-white-50"
                            ),
                            # Dropdown to control the bar chart
                            dcc.Dropdown(
                                id='dropdown_bar',
                                options=[
                                    {'label': x, 'value': x}
                                    for x in df['place'].fillna('Unknown').unique()
                                ],
                                multi=True,
                                value=['Dublin 6', 'Dun Laoghaire', 'Dublin 4',
                                       'Blackrock', 'Kinsale'],
                                optionHeight=35,
                            ),
                        ]),
                    dbc.Row(
                        className="my-5",  # border border-light 
                        children=[
                            dcc.Markdown(
                                'Enter a house attributes to get its predicted price',
                                className="text-center text-white-50 pt-2"
                            ),
                            # Allows the user to enter the place attribute of a house
                            dcc.Dropdown(
                                id='dropdown_place',
                                options=[
                                    {'label': x, 'value': x}
                                    for x in df['place'].fillna('Unknown').unique()
                                ],
                                multi=False,
                                className='pb-2',
                            ),
                            # Allows the user to enter the type_house attribute of a house
                            dcc.Dropdown(
                                id='dropdown_type_house',
                                options=[
                                    {'label': 'House', 'value': 'house'},
                                    {'label': 'Apartment', 'value': 'apartment'}
                                ],
                                multi=False,
                                className='pb-2',
                            ),
                            # Allows the user to enter the bedroom attribute of a house
                            dbc.Input(
                                id='input_bedrooms',
                                type='number',
                                placeholder='Bedrooms',
                                className='w-75 mx-auto text-center my-1 py-1',
                                min=1,
                                max=8,
                            ),
                            # Allows the user to enter the bathroom attribute of a house
                            dbc.Input(
                                id='input_bathrooms',
                                type='number',
                                placeholder='Bathrooms',
                                className='w-75 mx-auto text-center my-1 py-1 px-4',
                                min=1,
                                max=7,
                            ),
                            # Allows the user to enter the floor area attribute of a house
                            dbc.Input(
                                id='input_floor_area',
                                type='number',
                                placeholder='Floor Area',
                                className='w-75 mx-auto text-center my-1 py-1',
                                min=40,
                                max=625,
                            ),
                            # Allows the user to enter the latitude attribute of a house
                            dbc.Input(
                                id='input_latitude',
                                type='text',
                                placeholder='Latitude',
                                className='w-75 mx-auto text-center my-1 py-1',
                            ),
                            # Allows the user to enter the longitude attribute of a house
                            dbc.Input(
                                id='input_longitude',
                                type='text',
                                placeholder='Longitude',
                                className='w-75 mx-auto text-center my-1 px-2 bg-light py-1',
                            ),
                            # Button to submit the attributes entered
                            dbc.Button(
                                id='submit-button-state',
                                n_clicks=0,
                                children='Submit',
                                #className='w-75 mx-auto btn btn-dark px-2'
                                className='w-50 my-1 mx-auto btn btn-outline-success px-2'
                            ),
                            # This markdown returns the predicted price
                            dcc.Markdown(
                                id='markdown_price',
                                className="text-center my-2",   #text-white-50
                            #    style={'color': 'rgb(103,214,140)'}

                            ),
                        ]),
                    dcc.Markdown(
                        'CLICK ON A HOUSE AND GO TO ITS AD!!',
                        className="text-center",  #  pt-3    text-white-50
                        style={'color': 'rgb(216,173,173)'}
                    ),
                    # Link to the clicked house advertisement
                    html.Div([
                        html.A(
                            id='link_to_ad',
                        )])

                ],
                 #   className='pr-4',
                    width={'size': 2},
                    # style={"backgroundColor": "#100508"}
                )
            ],
            style={"height": "80vh"}
        ),
        # Third row (height 5)
        dbc.Row([
            # Empty column (width 8 of 12)
            dbc.Col(
                    width=8,
                    # style={"backgroundColor": "#100508"}
                    ),
            # Second column (width 4 of 12)
            dbc.Col(
                # Link to my LinkedIn ;)
                dcc.Link(
                    'LinkedIn',
                    href='https://www.linkedin.com/in/javier-casta%C3%B1o-candela-b89039208/',
                    className="text-center",
                    style={"backgroundColor": "#03182D", 'color': '#77C6FB'},
                    refresh=True,
                ),
                width=4,
                    ),
        ],
            style={"height": "5vh"}
        ),
    ],
    fluid=True,
    style={"height": "100vh",
           "background-color": "#03182D"}  #E3E3DC
)

# Callback section: Boostrap
# ------------------------------------------------------------------

#  By writing this decorator, we're telling Dash to call this function for us whenever
#  the value of the "input" component changes in order to update the children of
#  the "output" component on the page


@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='price_range_slider', component_property='value'),
    Input(component_id='floor_area_slider', component_property='value'),
    Input(component_id='residuals_range_slider', component_property='value'),
    Input(component_id='bedroom_range_slider', component_property='value'),
)
def update_output_map(price_range, floor_area_range, residuals_range, bedroom_range):
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
                    'url': False,
                    },
        hover_name='residual',
        opacity=1,
        template='plotly_dark'
    )
    map_fig.update(layout_coloraxis_showscale=False)

    return map_fig


@app.callback(
    Output('link_to_ad', 'children'),
    Output('link_to_ad', 'className'),
    Output('link_to_ad', 'href'),
    Output('link_to_ad', 'style'),
    Input('map', 'clickData'))
def display_click_data(clickData):
    # Get info from the click
    click_data = json.dumps(clickData, indent=2)
    # Take the url from the house clicked
    url = re.search(r'(https:).+', str(click_data))
    # When any house is clicked nothing is shown
    if url is None:
        url = ''
        className = "text-center text-light"
        href = ''
        style = {'color': '#77C6FB'}
    else:
        url = url.group().split('"')[0]
        className = "text-center font-weight-bold"  #text-white-50
        href = url
        style = {'color': '#77C6FB'}
    return url, className, href, style


@app.callback(
    Output('bar', 'figure'),
    Input('dropdown_bar', 'value'))
def display_click_data(places_list):
    # Mean price of the five places with higher mean prices according to the
    # filtered DataFrame
    mean_df = df.groupby('place')['actual_price'].mean() \
        .sort_values(ascending=False)
    if len(places_list) < 1:
        bar_fig = px.bar(
            mean_df,
            x=['Dublin 6', 'Dun Laoghaire', 'Dublin 4', 'Blackrock', 'Kinsale'],
            y=mean_df[['Dublin 6', 'Dun Laoghaire', 'Dublin 4', 'Blackrock',
                       'Kinsale']].values,
            labels={'x': '', 'y': ''},
            color_discrete_sequence=px.colors.sequential.RdBu,
            template='plotly_dark',
        )
    else:
        # Bar figure
        bar_fig = px.bar(
            mean_df,
            x=mean_df[places_list].index,
            y=mean_df[places_list].values,
            labels={'x': '', 'y': ''},
            color_discrete_sequence=px.colors.sequential.RdBu,
            template='plotly_dark',
        )

    return bar_fig


@app.callback(
    Output('markdown_price', 'children'),
    Output('markdown_price', 'style'),
    Input('submit-button-state', 'n_clicks'),
    State('dropdown_place', 'value'),
    State('dropdown_type_house', 'value'),
    State('input_bedrooms', 'value'),
    State('input_bathrooms', 'value'),
    State('input_floor_area', 'value'),
    State('input_latitude', 'value'),
    State('input_longitude', 'value'))
def display_click_data(n_clicks, place, type_house, bedrooms, bathrooms, floor_area,
                       latitude, longitude):

    # In case the user decide not enter the latitude or longitude
    if (latitude is None or longitude is None) and (place is not None):
        return 'Enter all the attributes please', {'color': 'rgb(190,10,10)'}
    # When the app first runs the input are NoneType
    if latitude is None:
        return 'Price prediction here!', {'color': 'rgb(103,214,140)'}
    # To check that the user enter valid data
    elif (50 < float(latitude[:2]) < 56) and (latitude[2] == '.'):
        latitude = float(latitude)
    else:
        return 'Enter a correct latitude', {'color': 'rgb(190,10,10)'}
    if longitude is None:
        return 'Price prediction here', {'color': 'rgb(103,214,140)'}
    # To check that the user enter valid data
    elif (longitude[2] == '.' or longitude[1] == '.') and (-11 < float(longitude) < 5.8):
        longitude = float(longitude)
    else:
        return 'Enter correct longitude', {'color': 'rgb(190,10,10)'}
    if (bedrooms is None) or (bathrooms is None) or (floor_area is None):
        return 'Enter all the attributes please', {'color': 'rgb(190,10,10)'}
    # Path to the model
    model_path = path.abspath(path.join('app.py',
                                        '../../investment-opportunities/data/06_models'
                                        '/final_model.pickle/2022-01-14T19.10.45.466Z'
                                        '/final_model.pickle'))
    # Load model
    model = joblib.load(model_path)

    # Transform the data entered to a DataFrame
    house = pd.DataFrame([[floor_area, latitude, longitude, bedrooms, bathrooms,
                           type_house, place]],
                         columns=['floor_area', 'latitude', 'longitude', 'bedroom',
                                  'bathroom', 'type_house', 'place'])
    # Predict the house price
    predicted_price = round(np.exp(model.predict(house)[0]))
    return f'{predicted_price}€', {'color': 'rgb(103,214,140)'}


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
