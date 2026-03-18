from tree_builder import *
from ONNX_builder import *

ONNX_PATH = "vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x4.onnx"

for i in range(10):
    root = c_max_prop(10, i+1)
    attach_logic_to_onnx(ONNX_PATH, root, f'{ONNX_PATH[:-5]}_ext{i}.onnx')