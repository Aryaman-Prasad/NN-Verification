from node_v2 import *
import numpy as np

def conjunct(leafnodes : list):
    return And(leafnodes)

def disjunct(leafnodes : list):
    return Or(leafnodes)

# y_i < y_c for all i from 1 to n (!= c)
def c_max_prop(n : int, c : int):

    # Creation of leaf nodes
    leafnodes = []
    for i in range(n):
        if i+1 == c:
            continue
        lin_map = np.zeros(n)
        lin_map[c-1] = 1
        lin_map[i] = -1
        lin_map = lin_map.astype(float)
        leafnode = Leaf(lin_map, 0.0)
        leafnodes.append(leafnode)

    return conjunct(leafnodes)

#y_i > y_c for all i from 1 to n (!= c)
def c_min_prop(n : int, c : int):

    # Creation of leaf nodes
    leafnodes = []
    for i in range(n):
        if i+1 == c:
            continue
        lin_map = np.zeros(n)
        lin_map[c-1] = -1
        lin_map[i] = 1
        lin_map = lin_map.astype(float)
        leafnode = Leaf(lin_map, 0.0)
        leafnodes.append(leafnode)

    return conjunct(leafnodes)

# CONF(N(x), N^(x)) < r, here r is threshold
def conf_under_r(n : int, r : float):
    delta = -np.log(100.0/r - 1) # r is assumed to be from 0 to 100

    disjunctnodes = []
    
    for i in range(n):
        leafnodes = []
        for j in range(n):
            if i == j:
                continue
            lin_map = np.zeros(n)
            lin_map[j] = 1
            lin_map[i] = -1
            lin_map = lin_map.astype(float)
            leafnode = Leaf(lin_map, delta)
            leafnodes.append(leafnode)

        disjunctnodes.append(disjunct(leafnodes))

    return conjunct(disjunctnodes)

# Relaxed robustness
def build_relaxed_robustness(n : int, c : int, r : float):
    prop1 = conf_under_r(n, r)

    prop2 = c_max_prop(n, c)

    return disjunct([prop1, prop2])