import numpy as np
import torch
from onnx2pytorch import ConvertModel
import onnx

# load adversarial input
x = np.load("adv_input.npy")

# load ONNX model
onnx_model = onnx.load("vnncomp2022_benchmarks/benchmarks/mnist_fc/onnx/mnist-net_256x2_ext3.onnx")
model = ConvertModel(onnx_model)

model.eval()

# run inference
x_tensor = torch.tensor(x, dtype=torch.float32)
output = model(x_tensor)

print("Network output:", output)