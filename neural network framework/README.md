# Neural Network Framework

A lightweight neural network framework built with NumPy for learning and experimentation.
It includes:
- autograd
- core tensor ops
- neural network modules
- optimizers
- dataset/dataloader utilities
- example training scripts

## Features

- `Tensor` with computation graph and backpropagation
- Core ops: add/mul/div/matmul, reshape, transpose, broadcast, sum, max
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `ELU`
- Attention:
  - `MultiHeadSelfAttention`
  - `SEBlock` (image-friendly channel attention)
- Loss op: `softmax_cross_entropy` (with backward)
- NN modules:
  - `Linear`, `Conv2d`, `Dropout`, `Flatten`, `Sequential`
  - `LayerNorm`, `TransformerEncoderBlock`, `SEBlock`
- Optimizers: `SGD`, `Adam`
- Data pipeline: `Dataset`, `DataLoader`
- Training examples:
  - `test_mnist.py`
  - `fashion_mnist.py`
  - `cifar10.py`
  - `attention_experiment.py`

## Quick Start

1. Install dependencies:
   - `pip install numpy pytest`
2. Run tests:
   - `pytest -q`
3. Run training scripts:
   - `python test_mnist.py --help`
   - `python test_mnist.py`
   - `python fashion_mnist.py`
   - `python cifar10.py`
   - `python attention_experiment.py`
4. CIFAR-10 model variants:
   - `python cifar10.py --model cnn`
   - `python cifar10.py --model cnn_se`
   - `python cifar10.py --model cnn_attn`
   - `python cifar10.py --model cnn_transformer`
   - `python cifar10.py --model vit`

## Project Structure

- `tensor.py`: tensor object, graph nodes, backward entry
- `autograd.py`: topological sort and gradient propagation
- `ops.py`: operator implementations and gradients
- `nn.py`: module system and common layers
- `optimizers.py`: optimizer implementations
- `data.py`: `Dataset` and `DataLoader`
- `tests/`: core/regression/activation tests
- `test_mnist.py`: MNIST training script
- `fashion_mnist.py`: Fashion-MNIST training script
- `cifar10.py`: CIFAR-10 training script
- `attention_experiment.py`: baseline vs attention comparison on a synthetic sequence task

## Notes

- Training scripts download datasets automatically when needed.
- Downloaded datasets are placed under:
  - `mnist_data/`
  - `fashion_mnist_data/`
  - `cifar10_data/`
- Attention effectiveness can be reproduced with:
  - `python attention_experiment.py`
  - default run prints final validation accuracy of baseline and attention models.
- For image tasks, `cnn_se` is usually a better attention baseline than pure transformer variants in this project.

## Suggested Next Modules / Functions

The following additions would provide the largest practical value:

1. Losses
- `mse_loss(pred, target)`
- `nll_loss(log_probs, target)`

2. Normalization Layers
- `BatchNorm1d`
- `BatchNorm2d`
- `LayerNorm`

3. Pooling Ops / Layers
- `max_pool2d`
- `avg_pool2d`

4. Regularization / Training Utilities
- `clip_grad_norm(parameters, max_norm)`
- `weight_decay` integration in optimizers
- learning rate schedulers (`StepLR`, `CosineAnnealingLR`)

5. Model Utilities
- `state_dict()` / `load_state_dict()`
- checkpoint save/load helpers

6. Data Utilities
- common transforms (`Normalize`, `RandomCrop`, `RandomHorizontalFlip`)
- deterministic dataloader seed support

7. Quality and Debug
- numerical gradient checker utility
- shape checker/assert helpers for debug mode
