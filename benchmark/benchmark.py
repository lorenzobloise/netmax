from data.transform_data import read_dataset
from im import IM
import pandas as pd





def benchmark_one_agent(file, algo, diffusion_model, budget=0, r=100):
    """
    Benchmark function to run the benchmarking for the given algorithm and diffusion model
    """
    # Load the graph
    g = read_dataset(file)
    # Delcare the agents dictionary
    if (budget == 0):
        # take the budget as the 10% of the number of nodes in the graph
        budget = int(0.1 * len(g.nodes))
    agents = dict({("Agent_01", budget)})
    # Create the im object
    im = IM(g, agents, algo, diffusion_model, r=r)

    # Run the benchmark
    seed = im.run()
    spread = im.result['spread']
    execution_time = im.result['execution_time']
    result_row = {
        "diffusion_model": [diffusion_model],
        "algorithm": [algo],
        "time": [execution_time],
        "seed": [seed],
        "spread": [spread]
    }
    # Take the results as a pandas dataframe
    df_results = pd.DataFrame(result_row)

    return df_results


def benchmark_budget_incr(file, algo, diffusion_model, budget=range(1, 100), r=100):
    """
    Benchmark function to run the benchmarking for the given algorithm and diffusion model
    """
    # Load the graph
    g = read_dataset(file)
    # Delcare the agents dictionary
    # Create the im object
    df_results = pd.DataFrame()
    for b in budget:
        agents = dict({("Agent_01", b)})
        im = IM(g, agents, algo, diffusion_model, r=r)
        # Run the benchmark
        seed = im.run()
        spread = im.result['spread']
        execution_time = im.result['execution_time']
        result_row = {
            "diffusion_model": [diffusion_model],
            "algorithm": [algo],
            "time": [execution_time],
            "seed": [seed],
            "spread": [spread],
            "budget": [b]
        }
        # Take the results as a pandas dataframe
        df_results = pd.concat([df_results, pd.DataFrame(result_row)], ignore_index=True)
    return df_results
