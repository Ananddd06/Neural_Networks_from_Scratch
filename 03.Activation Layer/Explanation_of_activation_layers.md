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
<img width="856" height="554" alt="Sigmoid-Activation-Function" src="https://github.com/user-attachments/assets/d294e738-c7ca-406b-890b-8950c9ff4ee6" />

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
<img width="976" height="576" alt="tanh" src="https://github.com/user-attachments/assets/3034655a-9ae4-4254-8e7c-4f7a5e7b56ad" />

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
<img width="846" height="554" alt="Relu-activation-function" src="https://github.com/user-attachments/assets/64411fc4-4924-4b1d-b03b-2ef80ef8d5e3" />

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
<img width="463" height="325" alt="leaky_relu" src="https://github.com/user-attachments/assets/1d271a26-3ec0-4cc6-8f67-0a5e525543a9" />

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
<img width="702" height="668" alt="1*Fw0mkoOfu45_WZ0RkHI45w" src="https://github.com/user-attachments/assets/1e106ce5-1fc5-437b-9132-5a45ecda7d89" />


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
