# NetMax - Influence Maximization in Social Networks

NetMax is a python library that provides the implementation of several algorithms for the problem of **Influence Maximization in Social Networks**, originally formulated in "Maximizing the Spread of Influence through a Social Network" (Kempe, Kleinberg and Tardos, 2003). NetMax is built upon NetworkX, a popular python library for working with graphs.

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
    [ ] Add partial seed set to algorithms
    [ ] Write a simulation method for competitive IM
    [ ] Define a score for each node computing:
        - The marginal gain for the current agent
        - The average marginal gain for the other agents
        - The average of the two previous

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

## Github links
    - https://github.com/snowgy/Influence_Maximization/blob/master/IMM.pdf (Paper)
