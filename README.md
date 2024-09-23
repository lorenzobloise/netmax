# NetMax - Influence Maximization in Social Networks

NetMax is a python library that provides the implementation of several algorithms for the problem of **Influence Maximization in Social Networks**, originally formulated in "Maximizing the Spread of Influence through a Social Network" (Kempe, Kleinberg and Tardos, 2003). NetMax is built upon NetworkX, a popular python library for working with graphs. It also addresses the problem of Competitive Influence Maximization, as an extensive-form strategic game setting in which multiple entities try to maximize their own influence across the network while minimizing the others'.

## Overview

## TODO
    [X] Implement a configuration object to store the parameters
    [X] Update CELF++ algorithm by computing mg2 from the second iteration instead of the first
    [X] Create "algorithms" directory and:
        [X] Create "simulation_based.py" with the code inside "algorithms.py"
        [X] Create "proxy_based.py" with the code inside "heuristics.py"
    [X] Implement Decreasing Cascade Diffusion Model
    [ ] Implement the proxy-based algorithms:
        [ ] IRIE
        [ ] SPIN
        [ ] IMRank
        [ ] SPM
        [ ] SP1M
        [ ] MIA
        [ ] PMIA
        [ ] IPA
        [ ] LDAG
    [X] Add partial seed set to algorithms
    [X] Write a simulation method for competitive IM
    [ ] Define a score for each node computing:
        - The marginal gain for the current agent
        - The average marginal gain for the other agents
        - The average of the two previous
    [ ] Clean up loggers and tqdms
    [ ] Fix UBLF and UBound problem with singular matrixes
    [ ] Define algorithms with a queue for each agent when using opinion-based dynamic influence probabilities
    [ ] Modify agents parameter inside competitive_influence_maximization.py
    [X] Delete single-agent directory
    [X] Fix multi-agent diffusion models:
        [X] Independent Cascade
        [X] Linear Threshold
        [X] Triggering
        [X] Decreasing Cascade
    [X] Add multi-processing from the following link:
        https://stackoverflow.com/questions/78510868/python-threads-do-not-utilize-cpu-cores-fully
    [ ] Parallelize the node exploration in simulation-based algorithms (use "multiprocessing" python library)
    [X] Add a graph visualization tool with Dash

## Ideas
    - Implement the sketch-based algorithms:
        [ ] NewGreIC
        [X] StaticGreedy
        [ ] PrunedMC
        [ ] SKIM
        [X] RIS
        [X] TIM
        [X] TIM+
        [ ] BKRIS
        [ ] SSA