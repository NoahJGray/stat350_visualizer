# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from plotly.subplots import make_subplots
from dash.dependencies import *
from symbulate import *
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import dash


app = dash.Dash(__name__)


max_bets=50


app.layout = html.Div(children=[
    html.H2(children='Joint Distribution Visualizer', style={'textAlign': 'center'}),
    html.Div(children=[
        html.Span(children=[
            html.B(children='Context: '),
            'Yolanda and Xavier are making bets on a roulette table and they have agreed to always make the same bet. However, the are not necessarily betting on every round! By using the visualizer below you can see the odds of them winning a particular number of rounds.'
        ]),
    ], style={'width': '75%', 'margin': 'auto'}),
    html.Div(children=[
        html.Div(children=[
            html.H3(children='Xavier\'s bets:'),
            html.Div([
                html.H3(id='x-low', style={'margin': '0'}),
                html.Div(children=[
                    dcc.RangeSlider(
                        id='x-slider',
                        min=1,
                        max=max_bets,
                        value=[1,max_bets],
                        updatemode='drag',
                        allowCross=False,
                        step=1
                    )
                ], style={'flex-grow': '1'}),
                html.H3(id='x-high', style={'margin': '0'})
            ], style={'display': 'flex'}),
        ], style={'width': '33%',}),
        html.Div(children=[
            html.H3(children='Type of bet:', style={'margin-bottom': '5px'}),
            dcc.Dropdown(id='bet-picker', options=[
                {'label': 'Straight', 'value': 1},
                {'label': 'Split', 'value': 2},
                {'label': 'Street', 'value': 3},
                {'label': 'Square', 'value': 4},
                {'label': 'Six Line', 'value': 6},
                {'label': 'Dozens', 'value': 12},
                {'label': 'Colors', 'value': 18}
            ], style={
                'width': '200px',
                'margin': 'auto',
            }, clearable=False, searchable=False, value=18),
        ], style={'width': '33%'}),
        html.Div(children=[
            html.H3(children='Yolanda\'s bets:'),
            html.Div([
                html.H3(id='y-low', style={'margin': '0'}),
                html.Div(children=[
                    dcc.RangeSlider(
                        id='y-slider',
                        min=1,
                        max=max_bets,
                        value=[1,max_bets],
                        updatemode='drag',
                        allowCross=False,
                        step=1
                    )
                ], style={'flex-grow': '1'}),
                html.H3(id='y-high', style={'margin': '0'})
            ], style={'display': 'flex'}),
        ], style={'width': '33%',})
    ], style={
        'display': 'flex',
        'justify-content': 'space-between'
    }),
    dcc.Tabs(id="tabs", value='pdf', children=[
        dcc.Tab(
            label='Probability Mass Function',
            value='pdf',
            style={'background-color': 'white'},
            selected_style={
                'border-top': '2px solid rgb(13,8,135)',
                'background-color': 'rgb(250,250,250)'
            }
        ),
        dcc.Tab(
            label='Cumulative Distribution Function',
            value='cdf',
            style={'background-color': 'white'},
            selected_style={
                'border-top': '2px solid rgb(13,8,135)',
                'background-color': 'rgb(250,250,250)'
            }
        ),
    ]),
    html.Div(children=[
        dcc.Graph(id='x-bar-graph'),
    ], style={
        'width': '33.33%',
        'display': 'inline-block',
    }),
    html.Div(children=[
        dcc.Graph(id='heatmap')
    ], style={
        'display': 'inline-block',
        'width': '33.33%'
    }),
    html.Div(children=[
        dcc.Graph(id='y-bar-graph'),
    ], style={
        'display': 'inline-block',
        'width': '33.33%'
    }),
    html.Span(children='By Noah Gray', style={
        'position': 'fixed',
        'bottom': '5px',
        'left': '5px',
        'margin': '0'
    })
])


@app.callback(
    Output('x-bar-graph', 'figure'),
    [Input('x-slider', 'value'), Input('bet-picker', 'value'), Input('tabs', 'value')]
)
def update_x_bar_graph(x_range, odds, tab):
    if tab == 'cdf':
        x = Binomial(n=(x_range[1]-x_range[0] + 1), p=odds/38).cdf(range(x_range[1]-x_range[0] + 2))
        x = [1] + [1 - prob for prob in x[:-1]]
    else:
        x = Binomial(n=(x_range[1]-x_range[0] + 1), p=odds/38).pmf(range(x_range[1]-x_range[0] + 2))
    fig = go.Figure([go.Bar(y=x, marker_color='rgb(13,8,135)')])
    fig.update_layout(
        title='Xavier\'s odds',
        yaxis=dict(
            title='Odds (decimal)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='Number of wins',
            titlefont_size=16,
            tickfont_size=14,
        ),
        paper_bgcolor='rgb(250,250,250)'
    )
    return fig


@app.callback(Output('x-low', 'children'), Input('x-slider', 'value'))
def update_x_low(x_range):
    return '(' + str(x_range[0]) + ')'


@app.callback(Output('x-high', 'children'), Input('x-slider', 'value'))
def update_x_low(x_range):
    return '(' + str(x_range[1]) + ')'


@app.callback(
    Output('y-bar-graph', 'figure'),
    [Input('y-slider', 'value'), Input('bet-picker', 'value'), Input('tabs', 'value')]
)
def update_y_bar_graph(y_range, odds, tab):
    if tab == 'cdf':
        y = Binomial(n=(y_range[1]-y_range[0] + 1), p=odds/38).cdf(range(y_range[1]-y_range[0] + 2))
        y = [1] + [1 - prob for prob in y[:-1]]
    else:
        y = Binomial(n=(y_range[1]-y_range[0] + 1), p=odds/38).pmf(range(y_range[1]-y_range[0] + 2))
    fig = go.Figure([go.Bar(y=y, marker_color='rgb(13,8,135)')])
    fig.update_layout(
        title='Yolanda\'s odds',
        yaxis=dict(
            title='Odds (decimal)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='Number of wins',
            titlefont_size=16,
            tickfont_size=14,
        ),
        paper_bgcolor='rgb(250,250,250)'
    )
    return fig


@app.callback(Output('y-low', 'children'), Input('y-slider', 'value'))
def update_x_low(y_range):
    return '(' + str(y_range[0]) + ')'


@app.callback(Output('y-high', 'children'), Input('y-slider', 'value'))
def update_x_low(y_range):
    return '(' + str(y_range[1]) + ')'


@app.callback(
    Output('heatmap', 'figure'),
    [Input('x-slider', 'value'),
    Input('y-slider', 'value'),
    Input('bet-picker', 'value'),
    Input('tabs', 'value')]
)
def update_heatmap(x_slider, y_slider, odds, tab):
    num_x = x_slider[1] - x_slider[0] + 1
    num_y = y_slider[1] - y_slider[0] + 1
    overlap = min(y_slider[1], x_slider[1]) - max(y_slider[0], x_slider[0]) + 1
    leftover = num_y - overlap
    if tab == 'cdf':
        x = Binomial(n=num_x, p=odds/38).cdf(range(num_x + 1))
        x = np.array([1] + [1 - prob for prob in x[:-1]])
        if overlap <= 0:
            y = Binomial(n=num_y, p=odds/38).cdf(range(num_y + 1))
            y = [[1] + [1 - prob for prob in y[:-1]]]*(num_x + 1)
        else:
            y = []
            for i in range(num_x + 1):
                temp = Binomial(n=leftover, p = odds/38).cdf(range(leftover + 1))
                temp = [1] + [1 - prob for prob in temp[:-1]]
                y.append([0]*min(i, overlap) + temp + [0]*(overlap - min(i, overlap)))
    else:
        x = np.array(Binomial(n=num_x, p=odds/38).pmf(range(num_x + 1)))
        if overlap <= 0:
            y = [list(Binomial(n=num_y, p=odds/38).pmf(range(num_y + 1)))]*(num_x + 1)
        else:
            y = []
            for i in range(num_x + 1):
                y.append([0]*min(i, overlap) + list(reversed(Binomial(n=leftover, p = odds/38).pmf(range(leftover + 1)))) + [0]*(overlap - min(i, overlap)))

    z = [list(np.array(y[i])*x[i]) for i in range(len(x))]

    fig = go.Figure([go.Heatmap(z=np.transpose(np.matrix(z)).tolist())])
    fig.update_layout(
        title='Combined odds (decimal)',
        yaxis=dict(
            title='Number of wins (Yolanda)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='Number of wins (Xavier)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        paper_bgcolor='rgb(250,250,250)'
    )

    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)

