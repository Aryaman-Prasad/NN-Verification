from tree_builder import *
from ONNX_builder import *


for i in range(10):
    root = c_min_prop(10, i+1)
    attach_logic_to_onnx('vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx', root, f'vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext{i}.onnx')