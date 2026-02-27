# Backpropagation in Neural Networks

---

## 1. What is Backpropagation?

**Backpropagation (Backward Propagation of Errors)** is an algorithm used to **train neural networks** by computing the **gradients of the loss function with respect to each weight and bias**.

In simple terms:

> Backpropagation tells us **how each parameter should change to reduce the error**.

---

## 2. Why Backpropagation is Needed

Neural networks learn by **minimizing a loss function**.

To minimize loss, we must know:

- Which parameters caused the error
- How strongly they contributed
- In which direction to update them

This information is provided by **gradients**:

```

∂Loss / ∂Parameter

```

👉 Backpropagation is the **efficient method** to compute these gradients.

---

## 3. What Happens Without Backpropagation?

Without backpropagation:

- We cannot compute gradients
- We cannot update weights correctly
- Loss does not decrease
- The network does not learn

Forward propagation alone only produces predictions.
**Learning happens only during backward propagation.**

---

## 4. Why Backpropagation is the Backbone of Neural Networks

Modern neural networks contain:

- Thousands to billions of parameters
- Multiple hidden layers
- Non-linear activation functions

Backpropagation is essential because it:

- Solves the **credit assignment problem** (who caused the error)
- Scales efficiently to deep networks
- Enables gradient-based optimization

Without backpropagation:

- Deep learning would not be practical
- CNNs, RNNs, Transformers would not work

Hence, backpropagation is called the **backbone of neural networks**.

---

## 5. High-Level Training Process

Training a neural network has two phases:

### 1. Forward Propagation

- Inputs are passed through the network
- Predictions are computed
- Loss is calculated

### 2. Backward Propagation

- Loss is propagated backward
- Gradients are computed using the chain rule
- Weights and biases are updated

---

## 6. Mathematical Foundation: Chain Rule

Backpropagation is based entirely on the **chain rule of calculus**.

For a weight `w`:

```

∂L/∂w =
(∂L/∂output)
× (∂output/∂activation)
× (∂activation/∂z)
× (∂z/∂w)

```

<img width="860" height="386" alt="chain_rule" src="https://github.com/user-attachments/assets/fd389cf7-bdcc-493d-98f3-aff9ebf737e1" />

Each term measures how changes propagate through the network.

---

## 7. Example Network (2 Inputs, 1 Hidden Neuron, Sigmoid)

### Network Structure

- Inputs: `x₁, x₂`
- Hidden neuron: weights `w₁, w₂`, bias `b₁`
- Output neuron: weight `w₃`, bias `b₂`
- Activation: Sigmoid
- Loss: Mean Squared Error (MSE)

---

## 8. Forward Propagation (Formulas)

### Hidden Layer

```

z₁ = w₁x₁ + w₂x₂ + b₁
a₁ = σ(z₁)

```

### Output Layer

```

z₂ = w₃a₁ + b₂
ŷ = σ(z₂)

```

---

## 9. Loss Function

Mean Squared Error:

```

L = ½ (y − ŷ)²

```

---

## 10. Backpropagation Derivation

### Output Layer Gradients

```

∂L/∂ŷ = ŷ − y
∂ŷ/∂z₂ = ŷ(1 − ŷ)
∂L/∂z₂ = (ŷ − y) · ŷ(1 − ŷ)

∂L/∂w₃ = ∂L/∂z₂ · a₁
∂L/∂b₂ = ∂L/∂z₂

```

---

### Hidden Layer Gradients

```

∂L/∂a₁ = ∂L/∂z₂ · w₃
∂a₁/∂z₁ = a₁(1 − a₁)
∂L/∂z₁ = (∂L/∂a₁)(∂a₁/∂z₁)

∂L/∂w₁ = ∂L/∂z₁ · x₁
∂L/∂w₂ = ∂L/∂z₁ · x₂
∂L/∂b₁ = ∂L/∂z₁

```

---

## 11. Parameter Update Rule

Using Gradient Descent:

```

w_new = w_old − η · ∂L/∂w
b_new = b_old − η · ∂L/∂b

```

Where `η` is the learning rate.

---

## 12. Why Backpropagation Works

- Gradients point in the direction of maximum error
- Subtracting gradients moves parameters toward minimum loss
- Repeated updates gradually improve predictions

This process continues until convergence.

---

## 13. Minimal Python Implementation (From Scratch)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Input and target
X = np.array([[0.5, 0.8]])
y = np.array([[1]])

# Initialize parameters
np.random.seed(1)
w1 = np.random.randn(2, 1)
b1 = np.random.randn(1)
w2 = np.random.randn(1, 1)
b2 = np.random.randn(1)

lr = 0.1

for _ in range(1000):
    # Forward pass
    z1 = X @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    y_hat = sigmoid(z2)

    # Loss
    loss = 0.5 * (y - y_hat) ** 2

    # Backward pass
    dL_dz2 = (y_hat - y) * sigmoid_derivative(y_hat)
    dL_dw2 = a1.T @ dL_dz2
    dL_db2 = dL_dz2

    dL_dz1 = (dL_dz2 @ w2.T) * sigmoid_derivative(a1)
    dL_dw1 = X.T @ dL_dz1
    dL_db1 = dL_dz1

    # Update
    w2 -= lr * dL_dw2
    b2 -= lr * dL_db2
    w1 -= lr * dL_dw1
    b1 -= lr * dL_db1

print("Final prediction:", y_hat)
```

---

## 14. Key Takeaways

- Backpropagation computes **exact gradients**
- It enables **learning in deep networks**
- Without it, neural networks cannot train
- It is fundamental to all modern deep learning systems

> **Backpropagation is not optional — it is essential.**
