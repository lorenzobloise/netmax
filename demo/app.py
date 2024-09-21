import dash
from dash import dcc, html
import plotly.graph_objs as go
from scipy.sparse import random
import random
from common.utils import read_adjacency_matrix
import multi_agent.competitive_influence_maximization as cim

# Initialize the Dash app
app = dash.Dash(__name__)


import networkx as nx

G = read_adjacency_matrix('../data/BigTestData.txt')
space_x, space_y = len(G.nodes()), len(G.nodes())
print(space_x, space_y)
pos={}
for node in G.nodes():
    pos[node] = (random.uniform(0, space_x), random.uniform(0, space_y))


edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=2,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))


# Define the layout of the app
app.layout = html.Div(style={'height': '100vh', 'width': '100vw'},children=[
    html.H1(children='Dash App Example'),

    html.Div(children='''
        This is a simple graph displayed using Dash.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                node_trace
            ],
            'layout': go.Layout(
                title='Simple Line Graph',
                xaxis={'title': 'X-axis Label'},
                yaxis={'title': 'Y-axis Label'},
                hovermode='closest'
            )
        },
        style={'height': '100%', 'width': '100%'}
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
