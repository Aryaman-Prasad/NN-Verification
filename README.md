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

Testing :
The vnncomp2022_benchmarks folder contains Neural Networks and properties to verify them on. As of now, main.py is used to take the NN and output the appended NN. The property will have to be slightly altered to work with the appended NN (since it only has a single output).

Run alpha-beta-crown as follows: (assuming you are in ./alpha-beta-CROWN/complete_verifier directory)
python abcrown.py --config ../../test.yaml

test.yaml is the config file for passing the NN and property along with additional parameters/flags

To run all benchmarks of a specific kind for a specific NN, run:
python test.py --original or python test.py --appended

Note that you will need to enter valid parameters inside the test.py code in order to run for various benchmarks and NN's, results will be stored in a csv