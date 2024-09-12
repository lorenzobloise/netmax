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
    [ ] Substitute tqdm with logger
    [ ] Fix UBLF problem with singular matrixes
    [ ] Define algorithms with a queue for each agent when using opinion-based dynamic influence probabilities
    [ ] Modify agents parameter inside competitive_influence_maximization.py
    [ ] Do in single-agent the same modifications done in multi-agent
    [ ] Fix multi-agent diffusion models
    [ ] Add multi-processing from the following link:
        https://stackoverflow.com/questions/78510868/python-threads-do-not-utilize-cpu-cores-fully

## Ideas
    - Add a Configuration object to IM
    - Implement the sketch-based algorithms:
        [ ] NewGreIC
        [ ] StaticGreedy
        [ ] PrunedMC
        [ ] SKIM
        [ ] RIS
        [ ] TIM
        [ ] TIM+
        [ ] BKRIS
        [ ] SSA
    - Add an opinion-based endorsement policy (maybe do not use it with opinion-based influence probability)
    - Add a graph visualization tool

## Github links
    - https://github.com/snowgy/Influence_Maximization/blob/master/IMM.pdf (Paper)
