# NN-Verification
Our tool for creating a circuit on top of a neural network to verify various robustness properties

Has two components:
1. tree_builder contains 
a. build_relaxed_robustness : this returns a binary tree for the logical formula required to check for relaxed robustness.

2. ONNX_builder contains
a. attach_logic_to_onnx : this  takes in the NN and attaches the given logical formula to it.

currently the final formula needs to be checked with >0 condition.

Idea : 
We implement 
1. AND between two expression by taking their min
2. OR between two expression by taking their max

The implementation works by first creating a logical formula in the form of a binary tree depending on the property we need to check . It then converts this logical formula into a set of ONNX layers which it attaches to the given model.