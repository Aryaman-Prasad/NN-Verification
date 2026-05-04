# class Node:
#     pass

# class And(Node):
#     def __init__(self,  left : Node , right : Node):
#         self.left = left
#         self.right = right


# class Or(Node):
#     def __init__(self, left : Node , right : Node):
#         self.left = left
#         self.right = right

# #lin_exp + bias > 0
# class Leaf(Node):
#     def __init__(self, lin_exp , bias):
#         self.lin_exp = lin_exp
#         self.bias = bias


class Node:
    _id_counter = 0
    def __init__(self):
        self.id = Node._id_counter
        Node._id_counter += 1

class And(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

class Or(Node):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

class Leaf(Node):
    def __init__(self, lin_exp, bias):
        super().__init__()
        self.lin_exp = lin_exp
        self.bias = bias
