from dash import Dash, dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
from utils import read_adjacency_matrix
import influence_maximization as im
import random

class Server:

    def __init__(self):
        self.app = Dash(__name__,external_stylesheets=['stylesheet.css'],suppress_callback_exceptions=True)
        dict_of_agents = self.__initialize_agents__()
        self.G = read_adjacency_matrix('../data/network.txt')
        self.im_instance  = im.InfluenceMaximization(input_graph=self.G, agents=dict_of_agents, alg='static_greedy',
                                                     diff_model='ic', inf_prob=None, r=100,
                                                     insert_opinion=False, endorsement_policy='random')
        self.G = self.im_instance.get_graph()
        self.agents = self.im_instance.get_agents()
        self.space_x, self.space_y, self.pos= self.__initialize_space__()
        self.node_trace, self.edge_trace = self.__initialize_traces__()
        self.fig = self.__initialize_figure__()
        self.app.layout = self.__initialize_layout__()
        self.game_played = False
        self.simulation_played = False
        self.colors = [
            'red',
            'blue',
            'green',
            'orange',
            'purple',
        ]

    def __initialize_layout__(self):
        button_style = {
            'backgroundColor': '#4CAF50',
            'color': 'white',
            'padding': '12px 24px',
            'border': 'none',
            'borderRadius': '4px',
            'margin': '10px',
            'cursor': 'pointer',
            'fontSize': '16px',
            'fontWeight': 'bold',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
            'transition': 'background-color 0.3s'
        }

        return html.Div(style={
            'minHeight': '100vh',
            'width': '100%',
            'padding': '20px',
            'backgroundColor': '#f5f5f5',
            'boxSizing': 'border-box',
            'overflow': 'hidden'
        }, children=[
            # Nuovo contenitore per la legenda in alto
            html.Div(style={
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'padding': '15px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'display': 'flex',
                'alignItems': 'center',
                'flexWrap': 'wrap',
                'gap': '20px'
            }, children=[
                # Titolo della legenda
                html.H4('Legend:', style={
                    'margin': '0',
                    'marginRight': '20px',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                }),
                # Container per gli elementi della legenda
                html.Div(style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'flexWrap': 'wrap',
                    'gap': '20px'
                }, children=[
                    # Inactive node
                    html.Div(style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '8px'
                    }, children=[
                        dcc.Graph(
                            figure={
                                'data': [{
                                    'x': [0],
                                    'y': [0],
                                    'mode': 'markers',
                                    'marker': {
                                        'symbol': 'circle',
                                        'size': 15,
                                        'color': '#808080'
                                    },
                                    'showlegend': False
                                }],
                                'layout': {
                                    'width': 30,
                                    'height': 30,
                                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                    'paper_bgcolor': 'rgba(0,0,0,0)',
                                    'plot_bgcolor': 'rgba(0,0,0,0)',
                                    'xaxis': {'visible': False},
                                    'yaxis': {'visible': False}
                                }
                            },
                            config={'displayModeBar': False},
                            style={'display': 'inline-block'}
                        ),
                        html.Span('Inactive node')
                    ]),
                    # Node activated
                    html.Div(style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '8px'
                    }, children=[
                        dcc.Graph(
                            figure={
                                'data': [{
                                    'x': [0],
                                    'y': [0],
                                    'mode': 'markers',
                                    'marker': {
                                        'symbol': 'circle',
                                        'size': 15,
                                        'color': '#1E90FF'
                                    },
                                    'showlegend': False
                                }],
                                'layout': {
                                    'width': 30,
                                    'height': 30,
                                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                    'paper_bgcolor': 'rgba(0,0,0,0)',
                                    'plot_bgcolor': 'rgba(0,0,0,0)',
                                    'xaxis': {'visible': False},
                                    'yaxis': {'visible': False}
                                }
                            },
                            config={'displayModeBar': False},
                            style={'display': 'inline-block'}
                        ),
                        html.Span('Node activated by agent blue')
                    ]),
                    # Seed set
                    html.Div(style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '8px'
                    }, children=[
                        dcc.Graph(
                            figure={
                                'data': [{
                                    'x': [0],
                                    'y': [0],
                                    'mode': 'markers',
                                    'marker': {
                                        'symbol': 'diamond',
                                        'size': 15,
                                        'color': '#1E90FF'
                                    },
                                    'showlegend': False
                                }],
                                'layout': {
                                    'width': 30,
                                    'height': 30,
                                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                    'paper_bgcolor': 'rgba(0,0,0,0)',
                                    'plot_bgcolor': 'rgba(0,0,0,0)',
                                    'xaxis': {'visible': False},
                                    'yaxis': {'visible': False}
                                }
                            },
                            config={'displayModeBar': False},
                            style={'display': 'inline-block'}
                        ),
                        html.Span('Node in the seed set of agent blue')
                    ]),
                    # Pending node
                    html.Div(style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '8px'
                    }, children=[
                        dcc.Graph(
                            figure={
                                'data': [{
                                    'x': [0],
                                    'y': [0],
                                    'mode': 'markers',
                                    'marker': {
                                        'symbol': 'hexagram',
                                        'size': 15,
                                        'color': '#808080'
                                    },
                                    'showlegend': False
                                }],
                                'layout': {
                                    'width': 30,
                                    'height': 30,
                                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                    'paper_bgcolor': 'rgba(0,0,0,0)',
                                    'plot_bgcolor': 'rgba(0,0,0,0)',
                                    'xaxis': {'visible': False},
                                    'yaxis': {'visible': False}
                                }
                            },
                            config={'displayModeBar': False},
                            style={'display': 'inline-block'}
                        ),
                        html.Span('Node pending for activation')
                    ])
                ])
            ]),

            # Resto del layout esistente...
            html.Div(style={
                'display': 'flex',
                'flexDirection': 'row',
                'flexWrap': 'wrap',
                'gap': '20px',
                'marginBottom': '20px',
                'minHeight': '70vh',
                'width': '100%'
            }, children=[
                # Contenitore del grafico
                html.Div(style={
                    'flex': '1 1 600px',
                    'minWidth': '300px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'boxSizing': 'border-box'
                }, children=[
                    dcc.Graph(
                        id='example-graph',
                        figure=self.fig,
                        style={'height': '100%', 'width': '100%'}
                    ),
                ]),

                # Tabella delle statistiche
                html.Div(style={
                    'flex': '1 1 300px',
                    'minWidth': '250px',
                    'maxWidth': '400px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'padding': '15px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'boxSizing': 'border-box'
                }, children=[
                    html.H3('Statistiche Partita', style={
                        'textAlign': 'center',
                        'marginBottom': '15px',
                        'color': '#333'
                    }),
                    html.Table(style={
                        'width': '100%',
                        'borderCollapse': 'collapse'
                    }, children=[
                        html.Tr([
                            html.Th('Metrica',
                                    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'left'}),
                            html.Th('Valore',
                                    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'right'})
                        ]),
                        html.Tr([
                            html.Td('Punteggio', style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td('0', id='score-value',
                                    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'right'})
                        ]),
                        html.Tr([
                            html.Td('Tempo', style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td('00:00', id='time-value',
                                    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'right'})
                        ]),
                        html.Tr([
                            html.Td('Mosse', style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                            html.Td('0', id='moves-value',
                                    style={'padding': '10px', 'borderBottom': '1px solid #ddd', 'textAlign': 'right'})
                        ])
                    ])
                ])
            ]),

            # Contenitore per slider (nascosto inizialmente)
            html.Div(id='slider-container',
                     hidden=True,
                     style={
                         'width': '80%',
                         'margin': '0 auto',
                         'maxWidth': '800px'
                     },
                     children=[
                         dcc.Slider(0, 20, 5, value=0, id='slider')
                     ]),

            # Contenitore per i pulsanti
            html.Div(style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'gap': '20px',
                'marginTop': '20px',
                'flexWrap': 'wrap'
            }, children=[
                html.Button(
                    'Start Game',
                    id='start-game-button',
                    n_clicks=0,
                    disabled=False,
                    style=button_style
                ),
                html.Button(
                    'Start Simulation',
                    id='start-simulation-button',
                    n_clicks=0,
                    style=button_style
                )
            ])
        ])



    def __initialize_agents__(self):
        agents = {}
        for i in range(3):
            agent_name = 'Agent_' + str(i)
            budget = random.randint(2, 2)
            agents[agent_name]=budget
        return agents

    def __initialize_figure__(self):
        fig = go.Figure(data=[self.node_trace, self.edge_trace],
                        layout=go.Layout(
                            title={
                                'text': '<b>NetMax Demo</b>',
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
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[],
            marker=dict(
                showscale=False,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                color=['#CCCCCC' for _ in range(len(self.G.nodes))],
                symbol=['circle' for _ in range(len(self.G.nodes))],
                size=25,
                line_width=[2 for _ in range(len(self.G.nodes))]))
        return node_trace, edge_trace

    def dispatcher_callback(self, n_clicks_game, n_clicks_simulation, slider_value):
        ctx = callback_context
        if not ctx.triggered:
            slider_container = self.app.layout.children[2].children
            hidden_value = self.app.layout.children[2].hidden
            return self.fig, slider_container, hidden_value,self._update_button_style(self.game_played)
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'start-game-button':
            fig = self.start_game(n_clicks_game)
            history = self.im_instance.get_history()
            slider = dcc.Slider(min=0, max=len(history)-1, step=1, value=len(history)-1, id='slider')
            self.app.layout.children[2].children = [slider]
            return fig, [slider] , False, self._update_button_style(self.game_played)
        elif button_id == 'start-simulation-button':
            self.clear_graph()
            fig = self.start_simulation(n_clicks_simulation)
            history = self.im_instance.get_diff_model().get_history()
            slider = dcc.Slider(min=0, max=len(history)-1, step=1, value=len(history)-1, id='slider')
            self.app.layout.children[2].children = [slider]
            self.simulation_played = True
            return fig, [slider], False,self._update_button_style(self.game_played)
        elif button_id == 'slider':
            # Handle slider value change if needed
            slider_container = self.app.layout.children[2].children
            slider_container[0].value = slider_value
            self.clear_graph()
            fig = self.__write_iteration__(slider_value)
            return fig, slider_container, False,self._update_button_style(self.game_played)
        return self.fig, [], False, self._update_button_style(self.game_played)

    def _update_button_style(self,game_is_already_started):
        if game_is_already_started:  # Sostituisci con la tua condizione
            style = {
                'backgroundColor': '#cccccc',
                'color': '#666666',
                'padding': '12px 24px',
                'border': 'none',
                'borderRadius': '4px',
                'margin': '10px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'cursor': 'not-allowed'
            }
            return style
        else:
            style = {
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'padding': '12px 24px',
                'border': 'none',
                'borderRadius': '4px',
                'margin': '10px',
                'cursor': 'pointer',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                'transition': 'background-color 0.3s'
            }
            return style

    def clear_graph(self):
        self.node_trace.marker.symbol = ['circle' for _ in range(len(self.G.nodes))]
        self.node_trace.marker.color = ['#CCCCCC' for _ in range(len(self.G.nodes))]

    def __write_iteration__(self, iteration):
        if not self.simulation_played:
            history=self.im_instance.get_history()
            dict_of_seed = {agent.name: agent.seed for agent in history[iteration]}
            self.fig = self.__update_figure__(dict_of_seed)
            return self.fig
        else:
            diff_model = self.im_instance.get_diff_model()
            history = diff_model.get_history()
            active_nodes,pending_nodes = history[iteration]
            self.fig = self.__write_simulation__(active_nodes, pending_nodes)
            return self.fig

    def __write_simulation__(self, active_nodes, pending_nodes):
        color = self.__assign_colors__(active_nodes)
        for agent_name, active_nodes in active_nodes.items():
            new_colors = tuple(color[agent_name] if i in active_nodes else self.node_trace.marker.color[i] for i in
                               range(len(self.G.nodes)))
            new_symbols = []
            for i in range(len(self.G.nodes)):
                if self.__is_in_some_seed_set__(i) and i in active_nodes:
                    new_symbols.append('diamond')
                elif i in pending_nodes:
                    new_symbols.append('hexagram')
                else:
                    new_symbols.append(self.node_trace.marker.symbol[i])
            self.node_trace.marker.color = new_colors
            self.node_trace.marker.symbol = new_symbols
        fig = go.Figure(data=[self.node_trace, self.edge_trace],
                        layout=go.Layout(
                            title={
                                'text': '<b>NetMax Demo</b>',
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

    def start_game(self, n_clicks):
        if(n_clicks==1):
            seed, _, _ = self.im_instance.run()
            history = self.im_instance.get_history()
            dict_of_seed = {agent.name: agent.seed for agent in history[len(history)-1]}
            self.fig = self.__update_figure__(dict_of_seed)
            self.game_played = True
            self.app.layout.children[2].disabled = True
            return self.fig
        return self.fig

    def start_simulation(self, n_clicks):
        diff_model = self.im_instance.get_diff_model()
        for agents in self.agents:
            agents.seed = [self.im_instance.mapping[x] for x in agents.seed]
        activates_nodes = diff_model.activate(self.G, self.agents)
        for agents in self.agents:
            agents.seed = [self.im_instance.inverse_mapping[x] for x in agents.seed]
        self.fig = self.__update_figure__(activates_nodes)
        self.simulation_played = True
        return self.fig

    def __assign_colors__(self, dict_agent_seed):
        colors = {}
        i = 0
        for agent_name,value in dict_agent_seed.items():
            colors[agent_name] = self.colors[i]
            i = (i + 1) % len(self.colors)
        return colors

    def __is_in_some_seed_set__(self, node):
        node = self.im_instance.inverse_mapping[node]
        for agent in self.agents:
            if node in agent.seed:
                return True
        return False

    def __update_figure__(self, seed_set_of_agents):
        color = self.__assign_colors__(seed_set_of_agents)
        for agent_name, active_nodes in seed_set_of_agents.items():
            new_colors = tuple(color[agent_name] if i in active_nodes else self.node_trace.marker.color[i] for i in range(len(self.G.nodes)))
            new_symbols = ['diamond' if (self.__is_in_some_seed_set__(i) and i in active_nodes) else self.node_trace.marker.symbol[i] for i in range(len(self.G.nodes))]
            self.node_trace.marker.color = new_colors
            self.node_trace.marker.symbol = new_symbols
        fig = go.Figure(data=[self.node_trace, self.edge_trace],
                        layout=go.Layout(
                            title={
                                'text': '<b>NetMax Demo</b>',
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
            Output('slider-container', 'children'),
            Output('slider-container', 'hidden'),
            Output('start-game-button', 'style'),
            Input('start-game-button', 'n_clicks'),
            Input('start-simulation-button', 'n_clicks'),
            Input('slider', 'value')
        )(self.dispatcher_callback)
        self.app.run(debug=True)