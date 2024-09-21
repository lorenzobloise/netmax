import dash
from dash import dcc, html
import plotly.graph_objs as go
import random
from common.utils import read_adjacency_matrix
from dash.dependencies import Input, Output
import multi_agent.competitive_influence_maximization as cim
from multi_agent.agent import Agent
import threading

# Initialize the Dash app
app = dash.Dash(__name__)

agents = []
for i in range(1):
    agent_name = 'Agent_' + str(i)
    agent = Agent(agent_name, random.randint(10,10))
    agent.__setattr__('id', i)
    agents.append(agent)

G = read_adjacency_matrix('../data/network.txt')
space_x, space_y = len(G.nodes()), len(G.nodes())

cim_instance = cim.CompetitiveInfluenceMaximization(input_graph=G, agents=agents, alg='mcgreedy',
                                                    diff_model='ic', inf_prob=None, r=1000,
                                                    insert_opinion=False, endorsement_policy='random')

pos = {}
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

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_text.append('# of connections: '+str(len(adjacencies[1])))

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        showscale=False,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        color=['#cccccc' for _ in range(len(G.nodes))],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

fig = go.Figure(data=[node_trace,edge_trace],
             layout=go.Layout(
                 title={
                     'text': '<b>NetMax Demo</b>',
                     'x': 0.5,
                     'xanchor': 'center'
                 },
                plot_bgcolor='white',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

# Define the layout of the app
app.layout = html.Div(style={'height': '100vh', 'width': '100vw'},children=[
    html.H1(children='NetMax Demo'),

    dcc.Graph(
        id='example-graph',
        figure=fig,
        style={'height': '100%', 'width': '100%'}
    ),

    dcc.Store(
        id='sim_graph',
        data=cim_instance.graph
    ),
])

# Callback to update the figure
@app.callback(
    Output('example-graph', 'figure'),
    Input('sim_graph', 'data')
)
def update_figure(data):
    active_nodes = [cim_instance.inverse_mapping(n) for n in cim.active_nodes(cim_instance.graph)]
    new_colors = tuple(['red' if i in active_nodes else node_trace.marker.color[i]])
    node_trace.marker.color = new_colors
    fig = go.Figure(data=[node_trace, edge_trace],
                    layout=go.Layout(
                        title={
                            'text': '<b>NetMax SCEMO</b>',
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        plot_bgcolor='white',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def start_game():
    cim_instance.run()

thread = threading.Thread(target=start_game)
thread.start()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)