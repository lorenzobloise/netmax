# NetMax - Influence Maximization in Social Networks

NetMax is a python library that provides the implementation of several algorithms for the problem of **Influence Maximization in Social Networks**, originally formulated in "Maximizing the Spread of Influence through a Social Network" (Kempe, Kleinberg and Tardos, 2003). NetMax is built upon NetworkX, a popular python library for working with graphs. It also addresses the problem of Competitive Influence Maximization, as an extensive-form strategic game setting in which multiple entities try to maximize their own influence across the network while minimizing the others'.

## Requirements

NetMax was developed with Python 3.12 and requires the installation of the following libraries:

- **networkx** (version 3.3)
- **numpy**
- **scipy**
- **tqdm**
- **heapdict**
- **pandas** (temporary)
- **dash** (version 2.18.0): useful for the demo of the framework

You can easily install all the requirements by running the following command:

`pip install -r requirements.txt`

## Overview

This framework wants to be a useful tool for all those people who study the problem of Influence Maximization. The users instantiate the `InfluenceMaximization` class, setting some basic parameters:

- `input_graph`: a directed graph representing the network (of type `networkx.DiGraph`)
- `agents`: a dictionary where the key is the agent name (`str`) and the value is his budget (`int`)
- `alg`: the algorithm to use for influence maximization (see [Algorithms](#algorithms))
- `diff_model`: the diffusion model to use (see [Diffusion models](#diffusion-models))
- `inf_prob`: the probability distribution used to generate (if needed) the probabilities of influence between nodes. The framework implements different influence probabilities, default is `None` (see [Influence probabilities](#influence-probabilities))
- `endorsement_policy`: the policy that nodes use to choose which agent to endorse when they have been contacted by more than one agent. The framework implements different endorsement policies, default is `'random'` (see [Endorsement policies](#endorsement-policies))
- `insert_opinion`: `True` if the nodes do not contain any information about their opinion on the agents, `False` otherwise (or if the opinion is not used)
- `inv_edges`: a `bool` indicating whether to invert the edges of the graph
- `r`: number of simulations to execute (default is 100)
- `verbose`: if `True` sets the logging level to `INFO`, otherwise displays only the minimal information

**Important**: `alg`, `diff_model`, `inf_prob` and `endorsement_policy` are `str` parameters, in order to prevent the user from directly importing and instantiating all the specific classes, which could have not been user-friendly.
If the user, after reading the documentation, wants to customize some specific parameters, can still change the corresponding attribute after the instantiation of the `InfluenceMaximization` object.
To view all the keywords for these parameters, see the corresponding section.

After creating the `InfluenceMaximization` object, the user may call its `run()` method, which returns:

- `seed`: a dictionary where the key is the agent name and the value is the seed set found
- `spread`: dictionary where the key is the agent name and the value is the expected spread
- `execution_time`: the total execution time (in seconds)

All these values are also available in the `result` attribute (which is a dictionary) of the `InfluenceMaximization` object.

### Algorithms

NetMax provides the implementation of many state-of-the-art algorithms. 

#### Simulation-based

- Monte-Carlo Greedy: implemented by the class `MCGreedy` (**keyword**: `mcgreedy`)
- CELF: implemented by the class `CELF` (**keyword**: `celf`)
- CELF++: implemented by the class `CELF_PP` (**keyword**: `celfpp`)

#### Proxy-based

- Highest Out-Degree Heuristic: implemented by the class `HighestOutDegree` (**keyword**: `outdeg`)
- Degree Discount: implemented by the class `DegDis` (**keyword**: `degdis`)
- Group PageRank: implemented by the class `Group_PR` (**keyword**: `group_pr`)

#### Sketch-based

- StaticGreedy: implemented by the class `HighestOutDegree` (**keyword**: `static_greedy`)
- RIS: implemented by the class `RIS` (**keyword**: `ris`)
- TIM: implemented by the class `TIM` (**keyword**: `tim`)
- TIM+: implemented by the class `TIMp` (**keyword**: `tim_p`)

### Diffusion models

The supported diffusion models are:

- Independent Cascade: implemented by the class `IndependentCascade` (**keyword**: `ic`)
- Linear Threshold: implemented by the class `LinearThreshold` (**keyword**: `lt`)
- Triggering Model: implemented by the class `Triggering` (**keyword**: `tr`)
- Decreasing Cascade: implemented by the class `DecreasingCascade` (**keyword**: `dc`)

### Influence probabilities

The influence probabilities are used to label the edges between the network nodes if they are not already labeled. The user can choose between:

- A constant value, set by default at `0.1` (**keyword**: `constant`)
- A uniform distribution between `0.01` and `0.1` (**keyword**: `uniform`)
- A distribution based on similarity between nodes computed with SimRank algorithm  (**keyword**: `similarity`)
- A ratio model which distributes the probability uniformly based on the in-degree of the target node (**keyword**: `ratio`)
- A hybrid approach based on the average degree of the graph (**keyword**: `hybrid`)
- An opinion-based approach (**keyword**: `opinion`) which assigns to each node a vector of **opinions** (namely, values between `0` and `1`) and computes the influence probability comparing the opinions of the two nodes
 with cosine similarity and taking into account also their SimRank similarity, with the formula:

$p(u,v)=b+k*\left(\frac{1}{outdeg(u)}*similarity(u,v)+cos\_sim(opinion(u),opinion(v))\right)$

### Endorsement policies

In the competitive setting it is possible that, in the same time step, multiple agents contact the same node. Therefore, it is necessary an endorsement policy that dictates which agent the
node chooses to endorse. Several endorsement policies are implemented:

- A random policy, which chooses randomly between the agents that contacted the node in that specific time step (**keyword**: `random`)
- A voting-based policy, which chooses the most occurring agent between the already activated neighbors of the node (**keyword**: `voting`)
- A community-based approach, which applies the voting strategy to the community the node belongs to instead of its neighbors (**keyword**: `community`)
- A similarity-based policy, which essentially is a weighted voting strategy based on the SimRank similarity between the node and its neighbors (**keyword**: `sim_endorsement`)