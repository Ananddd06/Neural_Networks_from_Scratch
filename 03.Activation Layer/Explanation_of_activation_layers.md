# ⚡ Activation Functions in Neural Networks (From Scratch)

---

## 🧠 What is an Activation Function?

An **activation function** decides **whether a neuron should be activated or not** by introducing **non-linearity** into the network.

👉 Without activation functions:

- Neural network becomes a **linear model**
- Cannot learn complex patterns ❌

---

## ❓ Why Do We Need Activation Functions?

They help to:
✅ Learn non-linear relationships  
✅ Control neuron output  
✅ Improve learning capacity  
✅ Enable deep networks

---

## 🔁 Where Are Activation Functions Used?

- **Hidden Layers** → Learn features
- **Output Layer** → Produce final prediction

---

# 🔥 Types of Activation Functions

---

## 1️⃣ Step Function (Threshold)

### 🧠 Idea

- Outputs only **0 or 1**
- Used in early perceptrons

### ❌ Problem

- Not differentiable
- Cannot use backpropagation

### 📌 Usage

- ❌ Not used in deep learning today

### 🧪 Python Code

```python
def step(x):
    return 1 if x >= 0 else 0
```

---

## 2️⃣ Sigmoid Function 🟢

### 📐 Formula

```text
σ(x) = 1 / (1 + e^(-x))
```

### 🧠 Why Use It?

- Converts output to **probability (0–1)**
- Smooth and differentiable

### 🚨 Problems Solved

- Binary classification output

### ❌ Problems

- Vanishing gradient
- Slow learning
- Output not zero-centered

### 📌 Where Used

- Output layer for **binary classification**

### 🧪 Python Code

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

---

## 3️⃣ Tanh Function 🔵

### 📐 Formula

```text
tanh(x) = (e^x − e^(−x)) / (e^x + e^(−x))
```

### 🧠 Why Use It?

- Output range **(-1, 1)**
- Zero-centered → faster convergence

### 🚨 Problems Solved

- Improves gradient flow compared to sigmoid

### ❌ Problems

- Still suffers from vanishing gradient

### 📌 Where Used

- Hidden layers (older architectures)

### 🧪 Python Code

```python
def tanh(x):
    return np.tanh(x)
```

---

## 4️⃣ ReLU (Rectified Linear Unit) 🔥

### 📐 Formula

```text
ReLU(x) = max(0, x)
```

### 🧠 Why Use It?

- Simple and fast
- Solves vanishing gradient problem
- Sparse activation

### 🚨 Problems Solved

- Enables deep networks to train efficiently

### ❌ Problems

- Dying ReLU (neurons stuck at 0)

### 📌 Where Used

- Hidden layers (default choice)

### 🧪 Python Code

```python
def relu(x):
    return np.maximum(0, x)
```

---

## 5️⃣ Leaky ReLU ⚡

### 🧠 Why Use It?

- Fixes **Dying ReLU** problem
- Allows small negative values

### 📐 Formula

```text
LeakyReLU(x) = x (x>0), 0.01x (x<0)
```

### 📌 Where Used

- Deep networks where ReLU fails

### 🧪 Python Code

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

---

## 6️⃣ ELU (Exponential Linear Unit) 🌊

### 🧠 Why Use It?

- Smooth negative region
- Faster convergence than ReLU

### ❌ Problem

- Computationally expensive

### 📌 Where Used

- Deep CNNs

### 🧪 Python Code

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha*(np.exp(x)-1))
```

---

## 7️⃣ Softmax 🔢

### 🧠 Why Use It?

- Converts logits into **class probabilities**
- Sum of outputs = 1

### 🚨 Problems Solved

- Multi-class classification

### 📌 Where Used

- Output layer for multi-class tasks

### 🧪 Python Code

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

---

## 🧠 Activation Function Selection Guide

| Situation                  | Activation |
| -------------------------- | ---------- |
| Hidden layers (default)    | ReLU       |
| Avoid dying neurons        | Leaky ReLU |
| Binary classification      | Sigmoid    |
| Multi-class classification | Softmax    |
| Older RNNs                 | Tanh       |

---
