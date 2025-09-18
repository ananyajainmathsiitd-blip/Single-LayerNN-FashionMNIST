# Neural Network Implementations: NumPy vs PyTorch  

This repository contains two implementations of a simple feedforward neural network trained on the **Fashion-MNIST** dataset:  

- **`numpynn.py`** – a from-scratch **NumPy implementation**, with manual forward/backward propagation and gradient updates.  
- **`pytorchnn.py`** – a **PyTorch implementation**, using `nn.Module`, autograd, and the DataLoader API.  

The purpose of this project is to compare how neural networks are built and trained **manually vs using a deep learning framework**.  

---

## Files  

- **`numpynn.py`**  
  - Loads Fashion-MNIST from CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`).  
  - Implements forward pass, ReLU activation, softmax output, and manual backpropagation.  
  - Uses **stochastic gradient descent (SGD)** with a fixed learning rate.  

- **`pytorchnn.py`**  
  - Loads Fashion-MNIST using `torchvision.datasets.FashionMNIST`.  
  - Defines a `torch.nn.Module` network with the same architecture as the NumPy version.  
  - Uses **CrossEntropyLoss** and **SGD optimizer** with automatic differentiation.  

---

## Network Architecture  

Both implementations use the same fully connected feedforward architecture:  

- **Input layer:** 784 (flattened 28×28 image)  
- **Hidden layer:** 64 units, activation **ReLU**  
- **Output layer:** 10 units (Fashion-MNIST classes), activation **softmax** (applied in NumPy; implicit in PyTorch’s CrossEntropyLoss)  

---

##  Hyperparameters  

| Parameter       | NumPy (`numpynn.py`) | PyTorch (`pytorchnn.py`) |
|-----------------|----------------------|--------------------------|
| Dataset         | Fashion-MNIST (CSV) | Fashion-MNIST (torchvision) |
| Epochs          | 40                  | 80                       |
| Batch size      | 1 (online SGD)      | 100                      |
| Learning rate   | 0.01                | 0.01                     |
| Optimizer       | Manual SGD          | torch.optim.SGD        |
| Loss function   | CrossEntropy (manual)| nn.CrossEntropyLoss    |

---

## Usage  

### Requirements  
- Python 3.8+  
- NumPy  
- PyTorch  
- torchvision (for PyTorch version)  
