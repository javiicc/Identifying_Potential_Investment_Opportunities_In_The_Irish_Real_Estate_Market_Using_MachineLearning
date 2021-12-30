dbc.Row([

    dbc.Col(html.H1("Ireland's Real Estate Market Opportunities",
                    className='pt-4 text-center text-primary, mb-4',
                    ),
            width=12)
],
    style={"height": "10vh"}  # , "backgroundColor": "black"
),
dbc.Row([
    dbc.Col(html.H6("Welcome to Javier Castaño Candela's final project from the "
                    "Master in Data Science at KSchool",
                    className='text-center'),
            width=12)
],
    style={"height": "5vh"}
),


'pt-4 text-center text-primary text-light, mb-4'



map_fig = px.scatter_mapbox(
    df,
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
              #  'url': True,
                },
    hover_name='residual',
    opacity=1,
    template='plotly_dark',
)


pie_fig = px.pie(
    df,
    values='actual_price',
    names='type_house',
    #title='',
    color_discrete_sequence=px.colors.sequential.RdBu,
    hover_data={'type_house': False,
                'actual_price': False},
    template='plotly_dark',
)

@app.callback(
   # Output(component_id='map', component_property='figure'),
    Output(component_id='pie', component_property='figure'),
    Input(component_id='price_range_slider', component_property='value'),
    Input(component_id='floor_area_slider', component_property='value'),
    Input(component_id='residuals_range_slider', component_property='value'),
    Input(component_id='bedroom_range_slider', component_property='value'),
    Input(component_id='bathroom_range_slider', component_property='value'),
)
def update_output_pie(price_range, floor_area_range, residuals_range, bedroom_range,
                      bathroom_range):
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
    pie_fig = px.pie(
        filtered_df,
        values='actual_price',
        names='type_house',
        # title='',
        color_discrete_sequence=px.colors.sequential.RdBu,
        hover_data={'type_house': False,
                    'actual_price': False},
        template='plotly_dark',
    )
    return pie_fig


'''
            dcc.Dropdown(
                id='city_dropdown',
                options=[{'label': x, 'value': x}
                         for x in df['place'].fillna('Unknown').unique()],
            #    value=list(df['place'].fillna('Unknown').unique()),
                multi=True
            ),
            '''