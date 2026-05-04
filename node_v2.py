import numpy as np

class Node:
    _id_counter = 0
    def __init__(self):
        self.id = Node._id_counter
        Node._id_counter += 1

class And(Node):
    def __init__(self, terms: list):
        super().__init__()
        self.terms = terms

class Or(Node):
    def __init__(self, terms: list):
        super().__init__()
        self.terms = terms

class Leaf(Node):
    def __init__(self, lin_exp, bias: float = 0.0):
        super().__init__()
        self.lin_exp = np.array(lin_exp, dtype=np.float32)
        self.bias = float(bias)