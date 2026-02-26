# Single Neuron: Forward Pass, Loss, and Backpropagation (Step‑by‑Step)


This document explains **exactly what is happening in the given diagram**, step by step, **mathematically and in NumPy code**.

We will cover:

1. Inputs, weights, and bias
2. Forward pass (multiplication → summation)
3. ReLU activation
4. Loss calculation
5. Backpropagation using **chain rule**
6. Full NumPy implementation with detailed explanation

---

## 1️⃣ Inputs, Weights, and Bias (from the diagram)

From the diagram:

| Symbol | Meaning       | Value |
| ------ | ------------- | ----- |
| x₀     | Input 0       | 1     |
| x₁     | Input 1       | -2    |
| x₂     | Input 2       | 3     |
| w₀     | Weight 0      | -3    |
| w₁     | Weight 1      | -1    |
| w₂     | Weight 2      | 2     |
| b      | Bias          | 1     |
| y      | Target output | 0     |

---

## 2️⃣ Forward Pass – Weighted Multiplication

Each input is multiplied by its weight:

[ x_0 w_0 = 1 \times (-3) = -3 ]
[ x_1 w_1 = (-2) \times (-1) = 2 ]
[ x_2 w_2 = 3 \times 2 = 6 ]

---

## 3️⃣ Summation + Bias

All weighted values are summed and bias is added:

[
Z = x_0 w_0 + x_1 w_1 + x_2 w_2 + b
]

[
Z = -3 + 2 + 6 + 1 = 6
]

This value **Z** is the neuron’s raw output (pre‑activation).

---

## 4️⃣ ReLU Activation Function

ReLU definition:

[
ReLU(Z) = \max(0, Z)
]

Since **Z = 6 > 0**:

[
A = ReLU(6) = 6
]

---

## 5️⃣ Loss Function (Squared Error)

Loss used in the diagram:

[
L = (A - y)^2
]

Substitute values:

[
L = (6 - 0)^2 = 36
]

This is the **error** produced by the neuron.

---

## 6️⃣ Backpropagation – Chain Rule

We compute gradients **backwards**.

### Step 1: Derivative of Loss w.r.t Activation

[
\frac{\partial L}{\partial A} = 2(A - y)
]

[
= 2(6 - 0) = 12
]

---

### Step 2: Derivative of ReLU

[
\frac{\partial A}{\partial Z} =
\begin{cases}
1 & Z > 0 \
0 & Z \le 0
\end{cases}
]

Since **Z = 6 > 0**:

[
\frac{\partial A}{\partial Z} = 1
]

---

### Step 3: Gradient of Z

[
Z = x_0 w_0 + x_1 w_1 + x_2 w_2 + b
]

So:

[
\frac{\partial Z}{\partial w_i} = x_i
]
[
\frac{\partial Z}{\partial b} = 1
]

---

### Step 4: Chain Rule (Final Gradients)

[
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial w_i}
]

#### Gradients:

[
\frac{\partial L}{\partial w_0} = 12 \times 1 \times 1 = 12
]
[
\frac{\partial L}{\partial w_1} = 12 \times 1 \times (-2) = -24
]
[
\frac{\partial L}{\partial w_2} = 12 \times 1 \times 3 = 36
]
[
\frac{\partial L}{\partial b} = 12
]

---

## 7️⃣ NumPy Implementation (Exact Diagram)

```python
import numpy as np

# Inputs
x = np.array([1, -2, 3])
w = np.array([-3, -1, 2])
b = 1
y = 0

# Forward pass
z = np.dot(x, w) + b

a = np.maximum(0, z)   # ReLU

loss = (a - y) ** 2

print("Z:", z)
print("Activation:", a)
print("Loss:", loss)

# Backpropagation

dL_dA = 2 * (a - y)
dA_dZ = 1 if z > 0 else 0

dL_dZ = dL_dA * dA_dZ

# Gradients

dL_dW = dL_dZ * x
dL_dB = dL_dZ

print("dL/dW:", dL_dW)
print("dL/dB:", dL_dB)
```

---

## 8️⃣ What You Learned

✔ How inputs multiply with weights
✔ How bias shifts the neuron
✔ Why ReLU blocks negative values
✔ How loss measures error
✔ How **chain rule** flows gradients backward
✔ How this becomes **gradient descent** in deep networks

---
