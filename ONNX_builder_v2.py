import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from collections import deque, defaultdict
from node_v2 import *

# ---------------------------------------------------------------------------
# Helpers
#
# Convention: all state tensors are row vectors of shape [1, n].
# Gemm(X, W, B, transB=1) computes X @ W.T + B:
#   [1, in] @ [out, in].T + [out]  =  [1, out]
# ---------------------------------------------------------------------------

def _gemm(tag, W, B, inp, node_list, initializers):
    """Emit: out[1,m] = inp[1,n] @ W[m,n].T + B[m]."""
    initializers.append(helper.make_tensor(f"W_{tag}", TensorProto.FLOAT, list(W.shape), W.flatten().tolist()))
    initializers.append(helper.make_tensor(f"B_{tag}", TensorProto.FLOAT, [W.shape[0]], B.flatten().tolist()))
    out = f"t_{tag}"
    node_list.append(helper.make_node("Gemm", [inp, f"W_{tag}", f"B_{tag}"], [out], transB=1))
    return out

def _relu(tag, inp, node_list):
    out = f"t_{tag}_relu"
    node_list.append(helper.make_node("Relu", [inp], [out]))
    return out


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _build_logic_layers(root, nn_output, node_list, initializers, eta):
    """
    Encode the property tree as a sequence of Gemm/ReLU layers.

    Leaf batch (4 Gemm + 2 ReLU total):
      Pass A:  r1  = ReLU( W_leaf @ y + b_leaf )      -- LE then ReLU
      Pass B1: r2  = ReLU( -r1 + eta )                -- eta - r1, then ReLU
      Pass B2: phi = (-1/eta)*r2 + 1                  -- 1 - r2/eta = phi

    Logic batch per depth level (3 Gemm + 1 ReLU):
      Gemm1: for each And/Or node build its pre-ReLU value; pass-through live values.
        AND row:  +1 per child, bias -(n_kids-1)   =>  sum(phi_i) - (n-1)
        OR  row:  -1 per child, bias +1            =>  1 - sum(phi_i)
      ReLU
      Gemm2: apply post-ReLU correction; pass-through live values.
        AND row:  +1, bias 0                       =>  identity (already phi_and)
        OR  row:  -1, bias +1                      =>  1 - ReLU(1-sum) = phi_or
      Shrink Gemm: compress the tensor to only live columns, updating the registry.

    The registry always maps node_id -> column index in the *current* tensor,
    and current_width == len(registry) is maintained as an invariant after each shrink.
    """
    eta = float(eta)

    # ------------------------------------------------------------------
    # 1. Topological analysis (Kahn's algorithm)
    # ------------------------------------------------------------------
    all_nodes = {}
    parents   = defaultdict(list)
    in_degree = {}
    ref_count = defaultdict(int)

    def collect(n):
        if n.id in all_nodes:
            return
        all_nodes[n.id] = n
        if isinstance(n, (And, Or)):
            in_degree[n.id] = len(n.terms)
            for c in n.terms:
                parents[c.id].append(n.id)
                ref_count[c.id] += 1
                collect(c)
        else:
            in_degree[n.id] = 0

    collect(root)
    ref_count[root.id] += 1  # root must survive to the end

    queue   = deque(nid for nid, d in in_degree.items() if d == 0)
    batches = []
    while queue:
        batch = list(queue); queue.clear()
        batches.append([all_nodes[nid] for nid in batch])
        for nid in batch:
            for pid in parents[nid]:
                in_degree[pid] -= 1
                if in_degree[pid] == 0:
                    queue.append(pid)

    leaves        = batches[0]
    logic_batches = batches[1:]

    # ------------------------------------------------------------------
    # 2. Leaf batch → phi values
    # ------------------------------------------------------------------
    n_L = len(leaves)

    # Pass A: LE = lin_exp @ y + bias, then ReLU  →  r1 shape [1, n_L]
    W_A = np.stack([l.lin_exp for l in leaves]).astype(np.float32)  # [n_L, input_dim]
    B_A = np.array([l.bias for l in leaves], dtype=np.float32)
    t   = _gemm("leaf_A",  W_A,  B_A, nn_output, node_list, initializers)
    t   = _relu("leaf_A",  t,  node_list)

    # Pass B1: eta - r1, then ReLU  →  r2 shape [1, n_L]
    W_B1 = -np.eye(n_L, dtype=np.float32)
    B_B1 =  np.full(n_L, eta, dtype=np.float32)
    t    = _gemm("leaf_B1", W_B1, B_B1, t, node_list, initializers)
    t    = _relu("leaf_B1", t, node_list)

    # Pass B2: 1 - r2/eta  →  phi shape [1, n_L]
    W_B2 = (-1.0 / eta) * np.eye(n_L, dtype=np.float32)
    B_B2 = np.ones(n_L, dtype=np.float32)
    t    = _gemm("leaf_B2", W_B2, B_B2, t, node_list, initializers)

    # registry: node_id -> column in t.  Invariant: current_width == len(registry).
    registry = {node.id: i for i, node in enumerate(leaves)}

    # ------------------------------------------------------------------
    # 3. Logic batches
    # ------------------------------------------------------------------
    for b_idx, batch in enumerate(logic_batches):
        prev_size = len(registry)   # == current tensor width (invariant)
        n_ops     = len(batch)

        # Nodes still needed now or later
        alive_ids = [nid for nid in registry if ref_count[nid] > 0]
        n_alive   = len(alive_ids)
        new_size  = n_ops + n_alive   # tensor width after Gemm2, before shrink

        # ---- Gemm1 + ReLU -------------------------------------------
        # Rows 0..n_ops-1: pre-ReLU values for each logic node
        # Rows n_ops..: pass-through of live predecessor values
        W1 = np.zeros((new_size, prev_size), dtype=np.float32)
        B1 = np.zeros(new_size, dtype=np.float32)

        for i, node in enumerate(batch):
            n_kids = len(node.terms)
            for child in node.terms:
                W1[i, registry[child.id]] = +1.0 if isinstance(node, And) else -1.0
            B1[i] = -(n_kids - 1) if isinstance(node, And) else +1.0

        for j, nid in enumerate(alive_ids):
            W1[n_ops + j, registry[nid]] = 1.0

        t = _gemm(f"lg{b_idx}_1", W1, B1, t, node_list, initializers)
        t = _relu(f"lg{b_idx}",   t,  node_list)

        # ---- Gemm2 --------------------------------------------------
        # Rows 0..n_ops-1: post-ReLU correction per logic node
        # Rows n_ops..: identity pass-through
        W2 = np.zeros((new_size, new_size), dtype=np.float32)
        B2 = np.zeros(new_size, dtype=np.float32)

        for i, node in enumerate(batch):
            if isinstance(node, And):
                W2[i, i] = +1.0          # already phi_and after ReLU
            else:
                W2[i, i] = -1.0          # negate ReLU(1 - sum)
                B2[i]    = +1.0          #   => 1 - ReLU(1 - sum) = phi_or

        for j in range(n_alive):
            W2[n_ops + j, n_ops + j] = 1.0

        t = _gemm(f"lg{b_idx}_2", W2, B2, t, node_list, initializers)

        # ---- Update ref counts --------------------------------------
        for node in batch:
            for child in node.terms:
                ref_count[child.id] -= 1

        # Positions in the current tensor (after Gemm2):
        #   node results: 0 .. n_ops-1
        #   alive predecessors: n_ops .. new_size-1
        pos_in_current = {}
        for i, node in enumerate(batch):
            pos_in_current[node.id] = i
        for j, nid in enumerate(alive_ids):
            pos_in_current[nid] = n_ops + j

        # ---- Shrink Gemm: compress to only still-live columns -------
        # This maintains the invariant current_width == len(registry).
        live_after = [nid for nid, cnt in ref_count.items()
                      if cnt > 0 and nid in pos_in_current]
        final_size = len(live_after)

        W_s = np.zeros((final_size, new_size), dtype=np.float32)
        new_registry = {}
        for new_col, nid in enumerate(live_after):
            W_s[new_col, pos_in_current[nid]] = 1.0
            new_registry[nid] = new_col

        t = _gemm(f"lg{b_idx}_s", W_s, np.zeros(final_size, dtype=np.float32),
                  t, node_list, initializers)
        registry = new_registry
        # Invariant restored: len(registry) == final_size == current tensor width ✓

    # ------------------------------------------------------------------
    # 4. Final selection: extract root's column as [1, 1]
    # ------------------------------------------------------------------
    n_live = len(registry)
    W_f    = np.zeros((1, n_live), dtype=np.float32)
    W_f[0, registry[root.id]] = 1.0
    return _gemm("final", W_f, np.zeros(1, dtype=np.float32), t, node_list, initializers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_logic_to_onnx(
    input_onnx_path: str,
    tree_root: Node,
    output_onnx_path: str,
    eta: float = 1e-3,
) -> None:
    """
    Appends phi-encoding property-verification layers to an ONNX model.

    All arithmetic is fused into Gemm/ReLU layers — no scalar Add/Sub/Mul nodes.
    Output is a [1,1] tensor: value 1 iff the property is satisfied (sound encoding).

    Parameters
    ----------
    input_onnx_path  : path to the original ONNX model.
    tree_root        : root of the property tree (And / Or / Leaf).
    output_onnx_path : path to save the extended ONNX model.
    eta              : phi threshold; smaller => narrower ambiguous zone [0, eta).
    """
    model = onnx.load(input_onnx_path)
    onnx.checker.check_model(model)

    if len(model.graph.output) != 1:
        raise ValueError(f"Expected 1 output, got {len(model.graph.output)}.")

    nn_out = model.graph.output[0].name
    new_nodes, new_inits = [], []

    result = _build_logic_layers(tree_root, nn_out, new_nodes, new_inits, eta=eta)

    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    model.graph.output.pop()
    model.graph.output.append(
        helper.make_tensor_value_info(result, TensorProto.FLOAT, [1, 1])
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_onnx_path)
    print(f"Extended model saved to: {output_onnx_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Base model: y = xW.T + b,  input [1,3] -> output [1,2]
    # y0 = x0 - x2,  y1 = x1 + x2
    W = np.array([[1.0, 0.0, -1.0],
                  [0.0, 1.0,  1.0]], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)

    x_in  = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y_out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    base_graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "W", "b"], ["y"], transB=1)],
        "linear", [x_in], [y_out],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
    )
    base_model = helper.make_model(base_graph, opset_imports=[helper.make_opsetid("", 17)])
    base_model.ir_version = 8
    onnx.checker.check_model(base_model)
    onnx.save(base_model, "./base_model.onnx")

    # Property: (y0 >= 0) AND ((y1 >= 0.5) OR (y0 - y1 >= 0))
    prop = And([
        Leaf(lin_exp=[1, 0], bias=0.0),       # y0 >= 0
        Or([
            Leaf(lin_exp=[0, 1], bias=-0.5),  # y1 >= 0.5
            Leaf(lin_exp=[1, -1], bias=0.0),  # y0 - y1 >= 0
        ])
    ])

    attach_logic_to_onnx("./base_model.onnx", prop, "./extended_model.onnx", eta=1e-3)

    base_sess = ort.InferenceSession("./base_model.onnx")
    ext_sess  = ort.InferenceSession("./extended_model.onnx")

    test_cases = [
        ("y0>=0, y1>=0.5 (all sat)",  np.array([[2.0,  1.0, -1.0]], dtype=np.float32)),
        ("y0<0 (AND fails)",          np.array([[-1.0, 1.0,  0.0]], dtype=np.float32)),
        ("y1<0.5 but y0-y1>=0 (OR)",  np.array([[1.0, -1.0,  0.0]], dtype=np.float32)),
        ("y0<0 and OR fails",         np.array([[-2.0,-1.0,  1.0]], dtype=np.float32)),
    ]

    print(f"\n{'Test case':<30} {'y0':>8} {'y1':>8} {'phi':>8}")
    print("-" * 58)
    for desc, x in test_cases:
        y   = base_sess.run(None, {"x": x})[0]
        phi = ext_sess.run(None,  {"x": x})[0]
        print(f"{desc:<30} {y[0,0]:>8.3f} {y[0,1]:>8.3f} {phi[0,0]:>8.3f}")

# import numpy as np
# import onnx
# import onnxruntime as ort
# from onnx import helper, TensorProto, numpy_helper
# from node_v2 import *

# # ---------------------------------------------------------------------------
# # ONNX graph builder
# # ---------------------------------------------------------------------------

# class _GraphBuilder:
#     """
#     Walks the property tree and emits ONNX nodes/initialisers, returning
#     the name of the tensor that holds the result of each sub-tree.
#     """

#     def __init__(self, eta: float):
#         self.eta = np.float32(eta)
#         self.nodes: list = []
#         self.initializers: list = []
#         self._counter = 0

#     def _uid(self, prefix: str) -> str:
#         self._counter += 1
#         return f"_logic_{prefix}_{self._counter}"

#     def _const_scalar(self, name: str, value: float) -> str:
#         arr = np.array(value, dtype=np.float32)
#         self.initializers.append(numpy_helper.from_array(arr, name=name))
#         return name

#     def _const_1d(self, name: str, arr: np.ndarray) -> str:
#         """Register a 1-D array constant, preserving its dtype."""
#         self.initializers.append(numpy_helper.from_array(arr, name=name))
#         return name

#     def _relu(self, x: str) -> str:
#         out = self._uid("relu")
#         self.nodes.append(helper.make_node("Relu", inputs=[x], outputs=[out]))
#         return out

#     def _add(self, a: str, b: str) -> str:
#         out = self._uid("add")
#         self.nodes.append(helper.make_node("Add", inputs=[a, b], outputs=[out]))
#         return out

#     def _sub(self, a: str, b: str) -> str:
#         out = self._uid("sub")
#         self.nodes.append(helper.make_node("Sub", inputs=[a, b], outputs=[out]))
#         return out

#     def _mul_scalar(self, x: str, value: float) -> str:
#         c_name = self._const_scalar(self._uid("scale"), value)
#         out = self._uid("mul")
#         self.nodes.append(helper.make_node("Mul", inputs=[x, c_name], outputs=[out]))
#         return out

#     def _gemv(self, x: str, weight: np.ndarray, bias: float) -> str:
#         """Computes weight @ x + bias, returning a scalar tensor of shape [1]."""
#         w = weight.reshape(1, -1).astype(np.float32)
#         w_name = self._uid("w")
#         self._const_1d(w_name, w.flatten())

#         shape_name = self._const_1d(self._uid("wshape"),
#                                     np.array([1, w.shape[1]], dtype=np.int64))
#         w_2d = self._uid("w2d")
#         self.nodes.append(helper.make_node("Reshape", inputs=[w_name, shape_name], outputs=[w_2d]))

#         x_shape_name = self._const_1d(self._uid("xshape"),
#                                       np.array([w.shape[1], 1], dtype=np.int64))
#         x_2d = self._uid("x2d")
#         self.nodes.append(helper.make_node("Reshape", inputs=[x, x_shape_name], outputs=[x_2d]))

#         mm_out = self._uid("mm")
#         self.nodes.append(helper.make_node("MatMul", inputs=[w_2d, x_2d], outputs=[mm_out]))

#         bias_name = self._const_scalar(self._uid("bias"), bias)
#         add_out = self._uid("gemv_out")
#         self.nodes.append(helper.make_node("Add", inputs=[mm_out, bias_name], outputs=[add_out]))

#         flat_shape = self._const_1d(self._uid("flat_shape"), np.array([1], dtype=np.int64))
#         flat_out = self._uid("flat")
#         self.nodes.append(helper.make_node("Reshape", inputs=[add_out, flat_shape], outputs=[flat_out]))
#         return flat_out

#     def _phi(self, le_tensor: str) -> str:
#         """Computes phi(LE) = 1 - (1/eta) * ReLU(eta - ReLU(LE)), returning a value in [0, 1]."""
#         relu_le   = self._relu(le_tensor)
#         eta_name  = self._const_scalar(self._uid("eta"), float(self.eta))
#         diff      = self._sub(eta_name, relu_le)
#         relu_diff = self._relu(diff)
#         scaled    = self._mul_scalar(relu_diff, float(1.0 / self.eta))
#         one_name  = self._const_scalar(self._uid("one"), 1.0)
#         return self._sub(one_name, scaled)

#     def _and(self, phi_tensors: list) -> str:
#         """Computes phi_AND = ReLU(sum(phi_i) - (n - 1))."""
#         if len(phi_tensors) == 1:
#             return phi_tensors[0]
#         running = phi_tensors[0]
#         for t in phi_tensors[1:]:
#             running = self._add(running, t)
#         n_minus_1 = self._const_scalar(self._uid("n_minus_1"), float(len(phi_tensors) - 1))
#         return self._relu(self._sub(running, n_minus_1))

#     def _or(self, phi_tensors: list) -> str:
#         """Computes phi_OR = 1 - ReLU(1 - sum(phi_i))."""
#         if len(phi_tensors) == 1:
#             return phi_tensors[0]
#         running = phi_tensors[0]
#         for t in phi_tensors[1:]:
#             running = self._add(running, t)
#         one_name  = self._const_scalar(self._uid("one"), 1.0)
#         diff      = self._sub(one_name, running)
#         relu_diff = self._relu(diff)
#         one_name2 = self._const_scalar(self._uid("one"), 1.0)
#         return self._sub(one_name2, relu_diff)

#     def build(self, node: Node, nn_output: str) -> str:
#         """Recursively walks the property tree, returning the result tensor name."""
#         if isinstance(node, Leaf):
#             return self._phi(self._gemv(nn_output, node.lin_exp, node.bias))
#         elif isinstance(node, And):
#             return self._and([self.build(child, nn_output) for child in node.terms])
#         elif isinstance(node, Or):
#             return self._or([self.build(child, nn_output) for child in node.terms])
#         else:
#             raise TypeError(f"Unknown node type: {type(node)}")


# # ---------------------------------------------------------------------------
# # Main entry point
# # ---------------------------------------------------------------------------

# def attach_logic_to_onnx(
#     input_onnx_path: str,
#     tree_root: Node,
#     output_onnx_path: str,
#     eta: float = 1e-3,
# ) -> None:
#     """
#     Appends property-verification layers to an ONNX model and saves the result.

#     The extended model retains the original inputs but produces a scalar output
#     in [0, 1]. Output = 1 means the property is satisfied (sound encoding);
#     output = 0 means unsatisfied or marginally satisfied (0 <= LE < eta).

#     Parameters
#     ----------
#     input_onnx_path  : path to the original ONNX model.
#     tree_root        : root of the property tree (And / Or / Leaf).
#     output_onnx_path : path to save the extended ONNX model.
#     eta              : ambiguity threshold; smaller => narrower ambiguous zone.
#     """
#     model = onnx.load(input_onnx_path)
#     onnx.checker.check_model(model)
#     graph = model.graph

#     if len(graph.output) != 1:
#         raise ValueError(
#             f"Expected a model with exactly 1 output, got {len(graph.output)}."
#         )

#     # Bridge the original output into the logic subgraph via an Identity node
#     original_output_name = graph.output[0].name
#     nn_out_internal = original_output_name + "_nn_raw"
#     identity_node = helper.make_node(
#         "Identity", inputs=[original_output_name], outputs=[nn_out_internal]
#     )

#     builder = _GraphBuilder(eta=eta)
#     result_tensor = builder.build(tree_root, nn_out_internal)

#     new_graph = helper.make_graph(
#         nodes=list(graph.node) + [identity_node] + builder.nodes,
#         name=graph.name + "_with_logic",
#         inputs=list(graph.input),
#         outputs=[helper.make_tensor_value_info(result_tensor, TensorProto.FLOAT, [1])],
#         initializer=list(graph.initializer) + builder.initializers,
#     )

#     new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
#     new_model.ir_version = model.ir_version
#     onnx.checker.check_model(new_model)
#     onnx.save(new_model, output_onnx_path)
#     print(f"Extended model saved to: {output_onnx_path}")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # --- Build a small example base model: linear map y = Wx + b ---
#     # 3 inputs -> 2 outputs, so y = [x0 - x2, x1 + x2]
#     W = np.array([[1.0, 0.0, -1.0],
#                   [0.0, 1.0,  1.0]], dtype=np.float32)
#     b = np.array([0.0, 0.0], dtype=np.float32)

#     x_input  = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
#     y_output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
#     base_graph = helper.make_graph(
#         [helper.make_node("MatMul", ["W", "x"], ["Wx"]),
#          helper.make_node("Add",    ["Wx", "b"], ["y"])],
#         "linear", [x_input], [y_output],
#         [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
#     )
#     base_model = helper.make_model(base_graph, opset_imports=[helper.make_opsetid("", 17)])
#     base_model.ir_version = 8
#     onnx.save(base_model, "./base_model.onnx")

#     # --- Define the property: (y0 >= 0) AND ((y1 >= 0.5) OR (y0 - y1 >= 0)) ---
#     prop = And([
#         Leaf(lin_exp=[1, 0], bias=0.0),        # y0 >= 0
#         Or([
#             Leaf(lin_exp=[0, 1], bias=-0.5),   # y1 >= 0.5
#             Leaf(lin_exp=[1, -1], bias=0.0),   # y0 - y1 >= 0
#         ])
#     ])

#     attach_logic_to_onnx("./base_model.onnx", prop, "./extended_model.onnx", eta=1e-3)

#     # --- Run and print results ---
#     base_sess = ort.InferenceSession("./base_model.onnx")
#     ext_sess  = ort.InferenceSession("./extended_model.onnx")

#     test_cases = [
#         ("y0>=0, y1>=0.5 (all sat)",  np.array([ 2.0,  1.0, -1.0], dtype=np.float32)),
#         ("y0<0 (AND fails)",          np.array([-1.0,  1.0,  0.0], dtype=np.float32)),
#         ("y1<0.5 but y0-y1>=0 (OR)",  np.array([ 0.05, -1.0,  0.0], dtype=np.float32)),
#         ("y0<0 and OR fails",         np.array([-2.0, -1.0,  1.0], dtype=np.float32)),
#     ]

#     print(f"\n{'Test case':<30} {'y0':>8} {'y1':>8} {'phi':>8}")
#     print("-" * 58)
#     for desc, x in test_cases:
#         y   = base_sess.run(None, {"x": x})[0]
#         phi = ext_sess.run(None,  {"x": x})[0]
#         print(f"{desc:<30} {y[0]:>8.3f} {y[1]:>8.3f} {phi[0]:>8.3f}")