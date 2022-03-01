"""
Port Pytorch Quickstart to NNI
==============================
This is a modified version of `Pytorch quickstart`_.

It can be run directly and will have the exact same result as original version.

Furthermore, it enables the ability of auto-tuning with an NNI *experiment*, which will be discussed later.

For now, we recommend to run this script directly to verify the environment.

There are only 3 key differences from the original version:

 1. In `Get optimized hyperparameters`_ part, it receives auto-generated hyperparameters.
 2. In `(Optional) Report intermediate results`_ part, it reports per-epoch accuracy for visualization.
 3. In `Report final result`_ part, it reports final accuracy for tuner to generate next hyperparameter set.

_Pytorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# %%
import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
# Hyperparameters to be tuned
# ---------------------------
params = {
    'linear_width': 512,
    'activation_type': 'relu',
    'learning_rate': 0.001,
    'momentum': 0.9
}

# %%
# Get optimized hyperparameters
# -----------------------------
# If run directly, ``nni.get_next_parameters()`` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

# %%
# Load dataset and define train/test function
# -------------------------------------------

from pathlib import Path
root_path = Path(__file__).parent.parent

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root=root_path / 'data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root=root_path / 'data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Train function.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test function.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct

# %%
# Build model and optimizer with hyperparameters
# ----------------------------------------------

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

activation_function_dict = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'tanh': nn.Tanh,
}

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_act_stack = nn.Sequential(
            nn.Linear(28 * 28, params['linear_width']),
            activation_function_dict[params['activation_type']](),
            nn.Linear(params['linear_width'], params['linear_width']),
            activation_function_dict[params['activation_type']](),
            nn.Linear(params['linear_width'], 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_act_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'])

# %%
# (Optional) Report intermediate results during training
# ------------------------------------------------------
# The callback reports per-epoch accuracy to show learning curve in NNI web portal.
# And in assessor tutorial (FIXME), you will see how to leverage the metrics for early stopping.
#
# You can safely skip ``nni.report_intermediate_result`` and the experiment will work as well.

epochs = 5
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    intermediate_accuracy = test(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(intermediate_accuracy)

# %%
# Report final result
# -------------------
# Report final accuracy to NNI so the tuning algorithm can 
nni.report_final_result(intermediate_accuracy)
