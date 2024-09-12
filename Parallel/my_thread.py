import threading
from tqdm import tqdm

# Step 1: Creare una classe che estende threading.Thread
class MyThread(threading.Thread):
    def __init__(self, diff_model, graph,agents,r,id):
        threading.Thread.__init__(self)
        self.diff_model = diff_model
        self.graph = graph
        self.agents = agents
        self.r = r
        self.result = None
        self.id = id

    def run(self):
        spreads = dict()
        for i in range(self.r):
            active_sets = self.diff_model.activate(self.graph,self.agents)
            for agent_name in active_sets.keys():
                spreads[agent_name] = spreads.get(agent_name, 0) + len(active_sets[agent_name])
        self.result= spreads

    def get_result(self):
        return self.result

#