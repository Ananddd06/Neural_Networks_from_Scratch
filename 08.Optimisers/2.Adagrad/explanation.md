# 📘 AdaGrad Optimization — In Depth (with NumPy Implementation)

---

## 1️⃣ Why Do We Even Need Better Optimizers?

In vanilla **Stochastic Gradient Descent (SGD)**:

$$
w_{t+1} = w_t - \eta \cdot \nabla L(w_t)
$$

- Same learning rate $\eta$ for **all parameters**
- Same learning rate **throughout training**
- Problems:
  - Some weights learn **too slowly**
  - Some weights **overshoot**
  - Sparse features learn very poorly

👉 This is where **adaptive learning rates** come in.

---

## 2️⃣ What Is AdaGrad?

**AdaGrad = Adaptive Gradient Algorithm**

Core idea:

> Parameters that receive **large gradients** should get **smaller learning rates**
> Parameters that receive **small / rare gradients** should get **larger learning rates**

---

## 3️⃣ The Key Intuition (Very Important)

AdaGrad **remembers past gradients**.

For each parameter $w_i$:

- If it has been updated **many times** → slow it down
- If it has been updated **rarely** → speed it up

This makes AdaGrad **excellent for sparse data**.

---

## 4️⃣ Mathematical Formulation

### Step 1: Accumulate squared gradients

$$
G_t = G_{t-1} + (\nabla L_t)^2
$$

- $G_t$ is **per-parameter**
- Squared gradients → always positive

---

### Step 2: Parameter update

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot \nabla L_t
$$

Where:

- $\eta$ = base learning rate
- $\epsilon$ = small value (e.g. $10^{-7}$) to avoid division by zero

---

## 5️⃣ Why AdaGrad Is Better Than Momentum (Conceptually)

### 🟢 Momentum

Momentum smooths updates:

$$
v_t = \beta v_{t-1} + \eta \nabla L_t
$$

✔ Faster convergence
❌ Same learning rate for all parameters
❌ Poor for sparse features

---

### 🔵 AdaGrad

| Feature                | AdaGrad      | Momentum |
| ---------------------- | ------------ | -------- |
| Adaptive LR per weight | ✅ Yes       | ❌ No    |
| Handles sparse data    | ✅ Excellent | ❌ Weak  |
| Requires tuning        | ❌ Less      | ✅ More  |
| Long-term LR decay     | ❌ Can stall | ✅ No    |

---

## 6️⃣ When AdaGrad Is BEST

✅ Sparse features (NLP, embeddings)
✅ Small datasets
✅ When you don’t want heavy tuning

❌ Very deep networks (learning rate may decay too much)

> 🔥 AdaGrad inspired **RMSProp** and **Adam**

---

## 7️⃣ Network Architecture (What We Build)

```
Input
  ↓
Dense (H1) + ReLU
  ↓
Dense (H2) + ReLU
  ↓
Dense (H3) + ReLU
  ↓
Dense (Output)
  ↓
MSE Loss
```

---

## 8️⃣ Full NumPy Code — 3 Hidden Layers + AdaGrad

```python
import numpy as np

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# Data
# -----------------------------
N = 64        # samples
D = 10        # input features
H1, H2, H3 = 32, 16, 8
O = 1

X = np.random.randn(N, D)
y = np.random.randn(N, O)

# -----------------------------
# Hyperparameters
# -----------------------------
lr = 0.1
epochs = 500
eps = 1e-7

# -----------------------------
# Initialize weights & biases
# -----------------------------
W1 = np.random.randn(D, H1) * 0.01
b1 = np.zeros((1, H1))

W2 = np.random.randn(H1, H2) * 0.01
b2 = np.zeros((1, H2))

W3 = np.random.randn(H2, H3) * 0.01
b3 = np.zeros((1, H3))

W4 = np.random.randn(H3, O) * 0.01
b4 = np.zeros((1, O))

# -----------------------------
# AdaGrad accumulators
# -----------------------------
GW1 = np.zeros_like(W1)
Gb1 = np.zeros_like(b1)

GW2 = np.zeros_like(W2)
Gb2 = np.zeros_like(b2)

GW3 = np.zeros_like(W3)
Gb3 = np.zeros_like(b3)

GW4 = np.zeros_like(W4)
Gb4 = np.zeros_like(b4)

# -----------------------------
# Activation
# -----------------------------
def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):

    # -------- Forward pass --------
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    a3 = relu(z3)

    y_pred = a3 @ W4 + b4

    # -------- Loss (MSE) --------
    loss = np.mean((y_pred - y) ** 2)

    # -------- Backward pass --------
    dy = 2 * (y_pred - y) / N

    dW4 = a3.T @ dy
    db4 = np.sum(dy, axis=0, keepdims=True)

    da3 = dy @ W4.T
    dz3 = relu_backward(da3, z3)

    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3 @ W3.T
    dz2 = relu_backward(da2, z2)

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = relu_backward(da1, z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # -------- AdaGrad update --------
    GW4 += dW4 ** 2
    Gb4 += db4 ** 2
    W4 -= lr * dW4 / (np.sqrt(GW4) + eps)
    b4 -= lr * db4 / (np.sqrt(Gb4) + eps)

    GW3 += dW3 ** 2
    Gb3 += db3 ** 2
    W3 -= lr * dW3 / (np.sqrt(GW3) + eps)
    b3 -= lr * db3 / (np.sqrt(Gb3) + eps)

    GW2 += dW2 ** 2
    Gb2 += db2 ** 2
    W2 -= lr * dW2 / (np.sqrt(GW2) + eps)
    b2 -= lr * db2 / (np.sqrt(Gb2) + eps)

    GW1 += dW1 ** 2
    Gb1 += db1 ** 2
    W1 -= lr * dW1 / (np.sqrt(GW1) + eps)
    b1 -= lr * db1 / (np.sqrt(Gb1) + eps)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# -----------------------------
# Final Results
# -----------------------------
print("\nFinal Loss:", loss)
print("W1 mean:", W1.mean(), "b1 mean:", b1.mean())
print("W4 mean:", W4.mean(), "b4 mean:", b4.mean())
```

---

## 9️⃣ What You Should Observe

✔ Loss steadily decreases
✔ Learning rate automatically adapts
✔ No momentum term needed
✔ Each parameter learns at its **own pace**

---

## 🔚 Final Takeaway

> **AdaGrad teaches us that learning rate is not a constant — it is information.**

Momentum accelerates,
AdaGrad **understands**.
