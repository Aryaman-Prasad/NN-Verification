from node import *
import numpy as np

# Conjunction
def conjunct(leafnodes : list):
    while True:
        newnodes = []
        for i in range(0, len(leafnodes), 2):
            if i+1 == len(leafnodes):
                newnodes.append(leafnodes[i])
                continue
            conjunct = And(left=leafnodes[i], right=leafnodes[i+1])
            newnodes.append(conjunct)
        if len(newnodes) == 1:
            return newnodes[0]
        leafnodes = newnodes

    assert(False)

# Disjunction
def disjunct(leafnodes):
    while True:
        newnodes = []
        for i in range(0, len(leafnodes), 2):
            if i+1 == len(leafnodes):
                newnodes.append(leafnodes[i])
                continue
            conjunct = Or(left=leafnodes[i], right=leafnodes[i+1])
            newnodes.append(conjunct)
        if len(newnodes) == 1:
            return newnodes[0]
        leafnodes = newnodes

    assert(False)

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

# CONF(N(x), N^(x)) < r, here r is threshold
def conf_under_r(n : int, r : float):
    delta = -np.log(100.0/r - 1) # r is assumed to be from 0 to 100

    disjunctnodes = []
    leafnodes = []
    for i in range(n):
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


# Idk why I made this
def print_tree(node, space=0, level_space=5):
    if type(node) == Leaf:
        space += level_space
        for i in range(len(node.lin_exp)):
            if node.lin_exp[i] == -1:
                print(" " * (space - level_space) + f"{node.lin_exp} | {node.bias}")
        return
    space += level_space
    print_tree(node.left, space)
    if type(node) == And:
        print(" " * (space - level_space) + f"and")
    elif type(node) == Or:
        print(" " * (space - level_space) + f"or")
    else:
        assert(False)
    print_tree(node.right, space)

# Testing
if __name__ == "__main__":
    label = 1
    num_outputs = 3
    threshold = 60.0
    assert(label >= 1 and label <= num_outputs)
    assert(threshold > 0 and threshold < 100)
    root = build_relaxed_robustness(num_outputs, label, threshold)
    print_tree(root)
