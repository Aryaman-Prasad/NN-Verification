import onnx
from onnx import helper, TensorProto
import numpy as np
from node import *


# --- 1. The Logic Builder ---

def recursive_builder(node, current_input, node_list, initializers, path, input_dim):
    match node:
        case And(left=l, right=r):
            left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
            right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

            diff = f"diff_{path}"
            relu_out = f"relu_{path}"
            output_name = f"and_out_{path}"

            # diff = right - left
            node_list.append(helper.make_node(
                'Sub', [right_out, left_out], [diff], name=f"Sub_{path}"
            ))

            # relu(diff)
            node_list.append(helper.make_node(
                'Relu', [diff], [relu_out], name=f"Relu_{path}"
            ))

            # min = right - relu(right-left)
            node_list.append(helper.make_node(
                'Sub', [right_out, relu_out], [output_name], name=f"MinViaRelu_{path}"
            ))

            return output_name

        case Or(left=l, right=r):
            left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
            right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

            diff = f"diff_{path}"
            relu_out = f"relu_{path}"
            output_name = f"or_out_{path}"

            # diff = right - left
            node_list.append(helper.make_node(
                'Sub', [right_out, left_out], [diff], name=f"Sub_{path}"
            ))

            # relu(diff)
            node_list.append(helper.make_node(
                'Relu', [diff], [relu_out], name=f"Relu_{path}"
            ))

            # max = left + relu(right-left)
            node_list.append(helper.make_node(
                'Add', [left_out, relu_out], [output_name], name=f"MaxViaRelu_{path}"
            ))

            return output_name

        case Leaf(lin_exp=weights, bias=b):

            w_name = f"W_{path}"
            b_name = f"B_{path}"

            matmul_out = f"matmul_{path}"
            leaf_out = f"leaf_out_{path}"

            weight_data = weights.astype(np.float32).reshape(input_dim, 1)

            initializers.append(helper.make_tensor(
                w_name, TensorProto.FLOAT, [input_dim, 1], weight_data.flatten()
            ))

            initializers.append(helper.make_tensor(
                b_name, TensorProto.FLOAT, [1], [float(b)]
            ))

            node_list.append(helper.make_node(
                "MatMul",
                [current_input, w_name],
                [matmul_out],
                name=f"MatMul_{path}"
            ))

            node_list.append(helper.make_node(
                "Add",
                [matmul_out, b_name],
                [leaf_out],
                name=f"Add_{path}"
            ))

            return leaf_out

# --- 2. The Stitching Function ---

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
    new_out_info = helper.make_tensor_value_info(final_name, TensorProto.FLOAT, [None, 1])
    model.graph.output.pop()
    model.graph.output.append(new_out_info)
    
    # Save combined model
    onnx.save(model, output_path)
    print(f"Combined model saved: {output_path}")