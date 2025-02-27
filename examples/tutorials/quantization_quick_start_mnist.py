"""
Quantization Quickstart
=======================

Quantization reduces model size and speeds up inference time by reducing the number of bits required to represent weights or activations.

In NNI, both post-training quantization algorithms and quantization-aware training algorithms are supported.
Here we use `QAT_Quantizer` as an example to show the usage of quantization in NNI.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-train on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Quantizing Model`_.

import torch
import torch.nn.functional as F
from torch.optim import SGD

from scripts.compression_mnist_model import TorchModel, trainer, evaluator, device

# define the model
model = TorchModel().to(device)

# define the optimizer and criterion for pre-training

optimizer = SGD(model.parameters(), 1e-2)
criterion = F.nll_loss

# pre-train and evaluate the model on MNIST dataset
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

# %%
# Quantizing Model
# ----------------
#
# Initialize a `config_list`.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.

config_list = [{
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_names': ['conv1']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_names': ['relu1']
}, {
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_names': ['conv2']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_names': ['relu2']
}]

# %%
# finetuning the model by using QAT
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
dummy_input = torch.rand(32, 1, 28, 28).to(device)
quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input)
quantizer.compress()
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

# %%
# export model and get calibration_config
model_path = "./log/mnist_model.pth"
calibration_path = "./log/mnist_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)

print("calibration_config: ", calibration_config)
