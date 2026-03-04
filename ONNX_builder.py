import onnx
from onnx import helper, TensorProto
import numpy as np

# --- 1. The Logic Tree Classes ---

class Node:
    pass

class And(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

class Or(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

class Leaf(Node):
    def __init__(self, lin_exp: np.ndarray, bias: float):
        self.lin_exp = lin_exp  # 1D numpy array: [feature_1, feature_2, ...]
        self.bias = bias        # Scalar float

# --- 2. The Logic Builder ---

def recursive_builder(node, current_input, node_list, initializers, path, input_dim):
    match node:
        case And(left=l, right=r):
            left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
            right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)
            
            output_name = f"and_out_{path}"
            # Min represents the intersection of bounds (AND)
            node_list.append(helper.make_node('Min', [left_out, right_out], [output_name], name=f"Min_{path}"))
            return output_name

        case Or(left=l, right=r):
            left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
            right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)
            
            output_name = f"or_out_{path}"
            # Max represents the union of bounds (OR)
            node_list.append(helper.make_node('Max', [left_out, right_out], [output_name], name=f"Max_{path}"))
            return output_name

        case Leaf(lin_exp=weights, bias=b):
            w_name = f"W_{path}"
            b_name = f"B_{path}"
            leaf_out = f"leaf_out_{path}"
            
            # Ensure weights are 2D for Gemm: [1, input_dim]
            # Since lin_exp is 1D [dim], we reshape to [1, dim]
            weight_data = weights.astype(np.float32).reshape(1, input_dim)
            
            initializers.append(helper.make_tensor(
                w_name, TensorProto.FLOAT, [1, input_dim], weight_data.flatten()
            ))
            initializers.append(helper.make_tensor(
                b_name, TensorProto.FLOAT, [1], [float(b)]
            ))
            
            # GEMM: Y = A * B + C 
            # where A is input [1, dim], B is weight [dim, 1], C is bias [1]
            # We use transB=1 to perform [1, dim] @ [1, dim].T + [1]
            node_list.append(helper.make_node(
                'Gemm',
                inputs=[current_input, w_name, b_name],
                outputs=[leaf_out],
                name=f"Gemm_Leaf_{path}",
                transB=1  # Transposes [1, dim] to [dim, 1] internally
            ))
            return leaf_out

# --- 3. The Stitching Function ---

def attach_logic_to_onnx(base_model_path, logic_tree, output_path):
    # Load the pre-existing model
    model = onnx.load(base_model_path)
    
    # Identify output of base model
    base_output = model.graph.output[0]
    base_output_name = base_output.name
    
    # Determine the feature dimension (e.g., 512 for a feature extractor)
    shape = base_output.type.tensor_type.shape.dim
    input_dim = shape[1].dim_value
    
    # Handle Flattening if coming from a Conv layer [Batch, C, H, W]
    current_input = base_output_name
    if len(shape) > 2:
        flatten_out = "logic_input_flattened"
        model.graph.node.append(helper.make_node('Flatten', [base_output_name], [flatten_out], axis=1))
        current_input = flatten_out

    # Build Logic Tree nodes
    new_nodes = []
    new_inits = []
    final_name = recursive_builder(logic_tree, current_input, new_nodes, new_inits, "logic", input_dim)
    
    # Merge into graph
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    
    # Update Model Output
    new_out_info = helper.make_tensor_value_info(final_name, TensorProto.FLOAT, [1, 1])
    model.graph.output.pop()
    model.graph.output.append(new_out_info)
    
    # Save combined model
    onnx.save(model, output_path)
    print(f"Combined model saved: {output_path}")