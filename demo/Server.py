from dash import Dash, dcc, html, Input, Output, callback_context

import plotly.graph_objs as go
from common.utils import read_adjacency_matrix
from multi_agent.agent import Agent
import multi_agent.competitive_influence_maximization as cim
import random

class Server:
    def __init__(self, input_graph):
        self.app = Dash(__name__)
        self.agents = self.__initialize_agents__()
        self.G= read_adjacency_matrix('../data/network.txt')
        self.cim_instance  = cim.CompetitiveInfluenceMaximization(input_graph=self.G, agents=self.agents, alg='tim',
                                                                diff_model='ic', inf_prob=None, r=100,
                                                                insert_opinion=False, endorsement_policy='random')
        self.space_x, self.space_y, self.pos= self.__initialize_space__()
        self.node_trace, self.edge_trace = self.__initialize_traces__()
        self.fig = self.__initialize_figure__()
        self.app.layout = self.__initialize_layout__()



    def __initialize_layout__(self):
        return html.Div(style={'height': '100vh', 'width': '100vw'}, children=[
            html.H1(children='NetMax Demo'),

            dcc.Graph(
                id='example-graph',
                figure=self.fig,
                style={'height': '100%', 'width': '100%'}
            ),

            html.Button('Start Game', id='start-game-button', n_clicks=0),

            html.Button('Start Simulation', id='start-simulation-button', n_clicks=0)


        ])

    def __initialize_agents__(self):
        agents = []
        for i in range(2):
            agent_name = 'Agent_' + str(i)
            agent = Agent(agent_name, random.randint(10, 10))
            agent.__setattr__('id', i)
            agents.append(agent)
        return agents

    def __initialize_figure__(self):
        fig = go.Figure(data=[self.node_trace, self.edge_trace],
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

    def __initialize_space__(self):
        space_x,space_y = len(self.G.nodes()), len(self.G.nodes())
        pos = {}
        for node in self.G.nodes():
            pos[node] = (random.uniform(0, space_x), random.uniform(0, space_y))
        return space_x, space_y, pos
    def __initialize_traces__(self):
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        node_x = []
        node_y = []
        for node in self.G.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_text.append('# of connections: ' + str(len(adjacencies[1])))

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
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                color=['#CCCCCC' for _ in range(len(self.G.nodes))],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        return node_trace, edge_trace


    def dispatcher_callback(self,n_clicks_game,n_clicks_simulation):
        ctx = callback_context
        if not ctx.triggered:
            return self.fig
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'start-game-button':
            return self.start_game(n_clicks_game)
        elif button_id == 'start-simulation-button' :
            self.clear_graph()
            return self.start_simulation(n_clicks_simulation)
        return self.fig

    def clear_graph(self):
        self.node_trace.marker.color = ['#CCCCCC' for _ in range(len(self.G.nodes))]



    def start_game(self, n_clicks):
        if(n_clicks==1):
            seed=self.cim_instance.run()
            print(seed)
            self.fig=self.__update_figure__(seed)
            return self.fig
        return self.fig

    def start_simulation(self, n_clicks):
        diff_model=self.cim_instance.get_diff_model()
        activates_nodes=diff_model.activate(self.G,self.agents)
        self.fig= self.__update_figure__(activates_nodes)
        return self.fig


    def __update_figure__(self,seed_set_of_agents):
        color={"Agent_0":'red',"Agent_1":'blue'}
        for agent_name,set in seed_set_of_agents.items():
            active_nodes = set
            new_colors = tuple(color[agent_name] if i in active_nodes else self.node_trace.marker.color[i] for i in range(len(self.G.nodes)))
            print(new_colors)
            self.node_trace.marker.color = new_colors
        fig = go.Figure(data=[self.node_trace, self.edge_trace],
                        layout=go.Layout(
                            title={
                                'text': '<b>NetMax Ciao</b>',
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


    def run(self):
        self.app.callback(
            Output('example-graph', 'figure'),
            Input('start-game-button', 'n_clicks'),
            Input('start-simulation-button', 'n_clicks')
        )(self.dispatcher_callback)
        self.app.run(debug=True)
