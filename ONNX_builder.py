import onnx
from onnx import helper, TensorProto
import numpy as np
from node import *
from collections import deque, defaultdict
# --- The Optimized Builder ---
def layer_by_layer_builder(root_node, base_input_name, node_list, initializers, input_dim):
    # 1. Dependency Analysis & Ref Counting
    adj = {} 
    in_degree = {}
    ref_count = defaultdict(int)
    all_nodes = {}

    def collect(n):
        if n.id in all_nodes: return
        all_nodes[n.id] = n
        if isinstance(n, (And, Or)):
            for child in [n.left, n.right]:
                adj.setdefault(child.id, []).append(n.id)
                in_degree[n.id] = in_degree.get(n.id, 0) + 1
                ref_count[child.id] += 1
                collect(child)
        else:
            in_degree[n.id] = 0
    
    collect(root_node)
    ref_count[root_node.id] += 1 

    # 2. Group into Batches (Kahn's Algorithm)
    queue = deque([n_id for n_id, deg in in_degree.items() if deg == 0])
    batches = []
    while queue:
        batch = []
        for _ in range(len(queue)):
            curr_id = queue.popleft()
            batch.append(all_nodes[curr_id])
            for parent_id in adj.get(curr_id, []):
                in_degree[parent_id] -= 1
                if in_degree[parent_id] == 0:
                    queue.append(parent_id)
        batches.append(batch)

    # 3. Batch 0 (Leaves)
    leaves = batches[0]
    W_leaf = np.stack([l.lin_exp for l in leaves])
    B_leaf = np.array([l.bias for l in leaves], dtype=np.float32)
    
    current_tensor = "t_b0_init"
    initializers.append(helper.make_tensor("W_b0", TensorProto.FLOAT, [len(leaves), input_dim], W_leaf))
    initializers.append(helper.make_tensor("B_b0", TensorProto.FLOAT, [len(leaves)], B_leaf))
    node_list.append(helper.make_node('Gemm', [base_input_name, "W_b0", "B_b0"], [current_tensor], transB=1))

    registry = {node.id: i for i, node in enumerate(leaves)}

    # 4. Process Logic Batches
    for b_idx, batch in enumerate(batches[1:], start=1):
        # FIX: We find which nodes are alive at the START of the batch.
        # This includes nodes used IN this batch.
        alive_at_start_ids = [nid for nid, idx in registry.items() if ref_count[nid] > 0]
        
        n_ops = len(batch)
        n_alive = len(alive_at_start_ids)
        new_size = n_ops + n_alive
        prev_size = len(registry)

        # --- Gemm 1: Diffs ---
        W1 = np.zeros((new_size, prev_size), dtype=np.float32)
        B1 = np.zeros(new_size, dtype=np.float32)
        
        for i, node in enumerate(batch):
            W1[i, registry[node.right.id]] = 1.0
            W1[i, registry[node.left.id]] = -1.0
        
        for i, nid in enumerate(alive_at_start_ids):
            W1[n_ops + i, registry[nid]] = 1.0

        t_diff = f"t_b{b_idx}_diff"
        initializers.append(helper.make_tensor(f"W_b{b_idx}_1", TensorProto.FLOAT, W1.shape, W1))
        initializers.append(helper.make_tensor(f"B_b{b_idx}_1", TensorProto.FLOAT, [new_size], B1))
        node_list.append(helper.make_node('Gemm', [current_tensor, f"W_b{b_idx}_1", f"B_b{b_idx}_1"], [t_diff], transB=1))

        node_list.append(helper.make_node('Relu', [t_diff], [f"t_b{b_idx}_relu"]))

        # --- Gemm 2: Final Logic & Buffer Compression ---
        W2 = np.zeros((new_size, new_size), dtype=np.float32)
        B2 = np.zeros(new_size, dtype=np.float32)

        for i, node in enumerate(batch):
            idx_relu = i
            idx_l_in_diff = n_ops + alive_at_start_ids.index(node.left.id)
            idx_r_in_diff = n_ops + alive_at_start_ids.index(node.right.id)

            if isinstance(node, And):
                W2[i, idx_r_in_diff] = 1.0
                W2[i, idx_relu] = -1.0
            else:
                W2[i, idx_l_in_diff] = 1.0
                W2[i, idx_relu] = 1.0

        for i in range(n_alive):
            W2[n_ops + i, n_ops + i] = 1.0

        current_tensor = f"t_b{b_idx}_out"
        initializers.append(helper.make_tensor(f"W_b{b_idx}_2", TensorProto.FLOAT, W2.shape, W2))
        initializers.append(helper.make_tensor(f"B_b{b_idx}_2", TensorProto.FLOAT, [new_size], B2))
        node_list.append(helper.make_node('Gemm', [f"t_b{b_idx}_relu", f"W_b{b_idx}_2", f"B_b{b_idx}_2"], [current_tensor], transB=1))

        # --- Update Ref Counts and Registry for NEXT batch ---
        for node in batch:
            ref_count[node.left.id] -= 1
            ref_count[node.right.id] -= 1

        # Only nodes still needed in FUTURE batches stay in the registry
        next_registry = {node.id: i for i, node in enumerate(batch) if ref_count[node.id] > 0}
        
        # Also carry forward nodes from the alive_at_start list that still have refs
        # We need a new compressed tensor to actually "shrink" the size
        alive_next_ids = [nid for nid in alive_at_start_ids if ref_count[nid] > 0]
        
        final_size = len(next_registry) + len(alive_next_ids)
        W_shrink = np.zeros((final_size, new_size), dtype=np.float32)
        
        # Map current positions to new compressed positions
        actual_next_registry = {}
        for i, (nid, _) in enumerate(next_registry.items()):
            W_shrink[i, registry_idx_in_current_out := i] = 1.0
            actual_next_registry[nid] = i
        
        curr_offset = len(next_registry)
        for i, nid in enumerate(alive_next_ids):
            # nid was at n_ops + original_index in the current_tensor
            old_pos = n_ops + alive_at_start_ids.index(nid)
            W_shrink[curr_offset + i, old_pos] = 1.0
            actual_next_registry[nid] = curr_offset + i
            
        shrink_tensor = f"t_b{b_idx}_shrunk"
        initializers.append(helper.make_tensor(f"W_b{b_idx}_s", TensorProto.FLOAT, W_shrink.shape, W_shrink))
        initializers.append(helper.make_tensor(f"B_b{b_idx}_s", TensorProto.FLOAT, [final_size], np.zeros(final_size, dtype=np.float32)))
        node_list.append(helper.make_node('Gemm', [current_tensor, f"W_b{b_idx}_s", f"B_b{b_idx}_s"], [shrink_tensor], transB=1))
        
        registry = actual_next_registry
        current_tensor = shrink_tensor

    # 5. Result Selection
    final_idx = registry[root_node.id]
    W_f = np.zeros((1, len(registry)), dtype=np.float32)
    W_f[0, final_idx] = 1.0
    initializers.append(helper.make_tensor("W_final", TensorProto.FLOAT, [1, len(registry)], W_f))
    initializers.append(helper.make_tensor("B_final", TensorProto.FLOAT, [1], [0.0]))
    node_list.append(helper.make_node('Gemm', [current_tensor, "W_final", "B_final"], ["logic_res"], transB=1))

    return "logic_res"

def attach_logic_to_onnx(base_model_path, logic_tree, output_path):
    model = onnx.load(base_model_path)
    base_out = model.graph.output[0].name
    dim = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    
    new_nodes, new_inits = [], []
    final_name = layer_by_layer_builder(logic_tree, base_out, new_nodes, new_inits, dim)
    
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    model.graph.output.pop()
    model.graph.output.append(helper.make_tensor_value_info(final_name, TensorProto.FLOAT, [1, 1]))
    
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

# import onnx
# from onnx import helper, TensorProto
# import numpy as np
# from node import *

# # --- Helper: create Gemm for linear combination ---
# def make_linear(node_list, initializers, inputs, coeffs, bias, name):

#     concat_out = f"{name}_concat"
#     node_list.append(helper.make_node(
#         'Concat',
#         inputs,
#         [concat_out],
#         axis=1,
#         name=f"{name}_concat_node"
#     ))

#     W_name = f"{name}_W"
#     b_name = f"{name}_b"
#     out_name = f"{name}_out"

#     W = np.array(coeffs, dtype=np.float32).reshape(1, -1)
#     b = np.array([bias], dtype=np.float32)

#     initializers.append(helper.make_tensor(
#         W_name, TensorProto.FLOAT, W.shape, W.flatten()
#     ))
#     initializers.append(helper.make_tensor(
#         b_name, TensorProto.FLOAT, [1], b
#     ))

#     node_list.append(helper.make_node(
#         'Gemm',
#         inputs=[concat_out, W_name, b_name],
#         outputs=[out_name],
#         name=f"{name}_gemm",
#         transB=1
#     ))

#     return out_name


# # --- Logic Builder ---
# def recursive_builder(node, current_input, node_list, initializers, path, input_dim):

#     match node:

#         # ================= AND (min)
#         case And(left=l, right=r):
#             left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
#             right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

#             # diff = right - left
#             diff = make_linear(
#                 node_list, initializers,
#                 [right_out, left_out],
#                 [1, -1],
#                 0.0,
#                 f"{path}_diff"
#             )

#             relu_out = f"{path}_relu"
#             node_list.append(helper.make_node(
#                 'Relu', [diff], [relu_out], name=f"{path}_relu_node"
#             ))

#             # min = right - relu
#             output_name = make_linear(
#                 node_list, initializers,
#                 [right_out, relu_out],
#                 [1, -1],
#                 0.0,
#                 f"{path}_min"
#             )

#             return output_name


#         # ================= OR (max)
#         case Or(left=l, right=r):
#             left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
#             right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

#             # diff = right - left
#             diff = make_linear(
#                 node_list, initializers,
#                 [right_out, left_out],
#                 [1, -1],
#                 0.0,
#                 f"{path}_diff"
#             )

#             relu_out = f"{path}_relu"
#             node_list.append(helper.make_node(
#                 'Relu', [diff], [relu_out], name=f"{path}_relu_node"
#             ))

#             # max = left + relu
#             output_name = make_linear(
#                 node_list, initializers,
#                 [left_out, relu_out],
#                 [1, 1],
#                 0.0,
#                 f"{path}_max"
#             )

#             return output_name


#         # ================= Leaf
#         case Leaf(lin_exp=weights, bias=b):
#             w_name = f"W_{path}"
#             b_name = f"B_{path}"
#             leaf_out = f"leaf_out_{path}"

#             weight_data = weights.astype(np.float32).reshape(1, input_dim)

#             initializers.append(helper.make_tensor(
#                 w_name, TensorProto.FLOAT, [1, input_dim], weight_data.flatten()
#             ))

#             initializers.append(helper.make_tensor(
#                 b_name, TensorProto.FLOAT, [1], [float(b)]
#             ))

#             node_list.append(helper.make_node(
#                 'Gemm',
#                 inputs=[current_input, w_name, b_name],
#                 outputs=[leaf_out],
#                 name=f"Gemm_Leaf_{path}",
#                 transB=1
#             ))

#             return leaf_out


# # --- Stitching Function (UNCHANGED) ---
# def attach_logic_to_onnx(base_model_path, logic_tree, output_path):
#     model = onnx.load(base_model_path)
    
#     base_output = model.graph.output[0]
#     base_output_name = base_output.name
    
#     shape = base_output.type.tensor_type.shape.dim
#     input_dim = shape[1].dim_value
    
#     current_input = base_output_name

#     if len(shape) > 2:
#         flatten_out = "logic_input_flattened"
#         model.graph.node.append(
#             helper.make_node('Flatten', [base_output_name], [flatten_out], axis=1)
#         )
#         current_input = flatten_out

#     new_nodes = []
#     new_inits = []

#     final_name = recursive_builder(
#         logic_tree,
#         current_input,
#         new_nodes,
#         new_inits,
#         "logic",
#         input_dim
#     )
    
#     model.graph.node.extend(new_nodes)
#     model.graph.initializer.extend(new_inits)
    
#     new_out_info = helper.make_tensor_value_info(
#         final_name, TensorProto.FLOAT, [1, 1]
#     )

#     model.graph.output.pop()
#     model.graph.output.append(new_out_info)
    
#     onnx.save(model, output_path)
#     print(f"Combined model saved: {output_path}")

# import onnx
# from onnx import helper, TensorProto
# import numpy as np
# from node import *

# # --- 1. The Logic Builder ---

# def recursive_builder(node, current_input, node_list, initializers, path, input_dim):
#     match node:
#         case And(left=l, right=r):
#             left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
#             right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

#             neg_left = f"neg_left_{path}"
#             diff = f"diff_{path}"
#             relu_out = f"relu_{path}"
#             neg_relu = f"neg_relu_{path}"
#             output_name = f"and_out_{path}"

#             minus_one_name = f"minus_one_{path}"
#             initializers.append(helper.make_tensor(
#                 minus_one_name, TensorProto.FLOAT, [1], [-1.0]
#             ))

#             # neg_left = -left
#             node_list.append(helper.make_node(
#                 'Mul', [left_out, minus_one_name], [neg_left], name=f"NegLeft_{path}"
#             ))

#             # diff = right + (-left)
#             node_list.append(helper.make_node(
#                 'Add', [right_out, neg_left], [diff], name=f"AddDiff_{path}"
#             ))

#             # relu(diff)
#             node_list.append(helper.make_node(
#                 'Relu', [diff], [relu_out], name=f"Relu_{path}"
#             ))

#             # neg_relu = -relu
#             node_list.append(helper.make_node(
#                 'Mul', [relu_out, minus_one_name], [neg_relu], name=f"NegRelu_{path}"
#             ))

#             # min = right + (-relu)
#             node_list.append(helper.make_node(
#                 'Add', [right_out, neg_relu], [output_name], name=f"MinViaRelu_{path}"
#             ))

#             return output_name

#         case Or(left=l, right=r):
#             left_out = recursive_builder(l, current_input, node_list, initializers, path + "_L", input_dim)
#             right_out = recursive_builder(r, current_input, node_list, initializers, path + "_R", input_dim)

#             neg_left = f"neg_left_{path}"
#             diff = f"diff_{path}"
#             relu_out = f"relu_{path}"
#             output_name = f"or_out_{path}"

#             minus_one_name = f"minus_one_{path}"
#             initializers.append(helper.make_tensor(
#                 minus_one_name, TensorProto.FLOAT, [1], [-1.0]
#             ))

#             # neg_left = -left
#             node_list.append(helper.make_node(
#                 'Mul', [left_out, minus_one_name], [neg_left], name=f"NegLeft_{path}"
#             ))

#             # diff = right + (-left)
#             node_list.append(helper.make_node(
#                 'Add', [right_out, neg_left], [diff], name=f"AddDiff_{path}"
#             ))

#             # relu(diff)
#             node_list.append(helper.make_node(
#                 'Relu', [diff], [relu_out], name=f"Relu_{path}"
#             ))

#             # max = left + relu
#             node_list.append(helper.make_node(
#                 'Add', [left_out, relu_out], [output_name], name=f"MaxViaRelu_{path}"
#             ))

#             return output_name

#         case Leaf(lin_exp=weights, bias=b):
#             w_name = f"W_{path}"
#             b_name = f"B_{path}"
#             leaf_out = f"leaf_out_{path}"

#             weight_data = weights.astype(np.float32).reshape(1, input_dim)

#             initializers.append(helper.make_tensor(
#                 w_name, TensorProto.FLOAT, [1, input_dim], weight_data.flatten()
#             ))

#             initializers.append(helper.make_tensor(
#                 b_name, TensorProto.FLOAT, [1], [float(b)]
#             ))

#             node_list.append(helper.make_node(
#                 'Gemm',
#                 inputs=[current_input, w_name, b_name],
#                 outputs=[leaf_out],
#                 name=f"Gemm_Leaf_{path}",
#                 transB=1
#             ))

#             return leaf_out


# # --- 2. The Stitching Function ---

# def attach_logic_to_onnx(base_model_path, logic_tree, output_path):
#     model = onnx.load(base_model_path)
    
#     base_output = model.graph.output[0]
#     base_output_name = base_output.name
    
#     shape = base_output.type.tensor_type.shape.dim
#     input_dim = shape[1].dim_value
    
#     current_input = base_output_name

#     # Flatten if needed
#     if len(shape) > 2:
#         flatten_out = "logic_input_flattened"
#         model.graph.node.append(
#             helper.make_node('Flatten', [base_output_name], [flatten_out], axis=1)
#         )
#         current_input = flatten_out

#     # Build logic
#     new_nodes = []
#     new_inits = []

#     final_name = recursive_builder(
#         logic_tree,
#         current_input,
#         new_nodes,
#         new_inits,
#         "logic",
#         input_dim
#     )
    
#     # Merge
#     model.graph.node.extend(new_nodes)
#     model.graph.initializer.extend(new_inits)
    
#     # Replace output
#     new_out_info = helper.make_tensor_value_info(
#         final_name, TensorProto.FLOAT, [1, 1]
#     )

#     model.graph.output.pop()
#     model.graph.output.append(new_out_info)
    
#     # Save
#     onnx.save(model, output_path)
#     print(f"Combined model saved: {output_path}")