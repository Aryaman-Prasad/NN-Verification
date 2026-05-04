# NN-Verification
Our tool for creating a circuit on top of a neural network to verify various robustness properties. Developed by Aryaman Prasad (myself) and Anshul Goyal

Has two components:
1. tree_builder contains 
a. build_relaxed_robustness : this returns a binary tree for the logical formula required to check for relaxed robustness.

2. ONNX_builder contains
a. attach_logic_to_onnx : this  takes in the NN and attaches the given logical formula to it.

currently the final formula needs to be checked with >0 condition.

1st Idea : 
We implement 
1. AND between two expression by taking their min
2. OR between two expression by taking their max

The implementation works by first creating a logical formula in the form of a binary tree depending on the property we need to check . It then converts this logical formula into a set of ONNX layers which it attaches to the given model.

2nd Idea :
Convert each linear expression into a boolean signal via two layers of ReLUs, followed by conjunctions/disjunctions of the booleans in an appropriate manner

The implementation for this follows a similar pipeline as the 1st idea, with the relevant files having a `_v2` suffix in their name

Testing :
The vnncomp2022_benchmarks folder contains Neural Networks and properties to verify them on. As of now, main.py is used to take the NN and output the appended NN. The property will have to be slightly altered to work with the appended NN (since it only has a single output).

Run alpha-beta-crown as follows: (assuming you are in ./alpha-beta-CROWN/complete_verifier directory)
`python abcrown.py --config ../../test.yaml`

test.yaml is the config file for passing the NN and property along with additional parameters/flags

To run all benchmarks of a specific kind for a specific NN, run:
`python test.py --original` or `python test.py --appended`

In case of marabou, run `test_marabou.py` instead of `test.py` (you will need marabou installed for this though)

Note that you will need to enter valid parameters inside the test.py code in order to run for various benchmarks and NN's, results will be stored in a csv, you may also need to comment/uncomment a line in the `create_alt_property` function depending on the type of encoding

The results and corresponding cactus plots for a bunch of tests conducted by us can be found in the `results/` and `plots/` directories