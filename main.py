from tree_builder import *
from ONNX_builder import *

root = c_min_prop(10, 8)
attach_logic_to_onnx('vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx/mnist-net_256x2.onnx', root, 'vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2.onnx/test.onnx')