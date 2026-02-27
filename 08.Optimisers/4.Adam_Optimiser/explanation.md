# 📘 Adam Optimizer — Deep Explanation + NumPy Code

---

## 1️⃣ Why Adam Was Invented

Let’s recap the problems:

### SGD

❌ Same learning rate for all parameters
❌ Very noisy updates

### Momentum

✔ Faster convergence
❌ Still same learning rate for all parameters

### AdaGrad

✔ Adaptive learning rate per parameter
❌ Learning rate keeps shrinking → training can stall

### RMSProp

✔ Fixes AdaGrad’s shrinking LR
❌ No momentum on gradients

👉 **Adam fixes ALL of these together**

---

## 2️⃣ What Adam Really Is (Big Picture)

**Adam = Adaptive Moment Estimation**

Adam keeps track of **two things** for every parameter:

1. **First moment (mean of gradients)** → like **Momentum**
2. **Second moment (variance of gradients)** → like **RMSProp**

So Adam knows:

- **Direction** to move (momentum)
- **How big** the step should be (adaptive LR)

---

## 3️⃣ Adam Mathematics (Step by Step)

Let gradient at time $t$ be:

$$
g_t = \nabla L(w_t)
$$

---

### 🔹 Step 1: First moment (Momentum)

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

- Smooths gradients
- Reduces noise
- Typical value: $\beta_1 = 0.9$

---

### 🔹 Step 2: Second moment (RMSProp-like)

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

- Tracks gradient magnitude
- Controls step size
- Typical value: $\beta_2 = 0.999$

---

### 🔹 Step 3: Bias correction (VERY IMPORTANT)

Early in training, $m_t$ and $v_t$ are biased toward zero.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

---

### 🔹 Step 4: Parameter update

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

---

## 4️⃣ Key Intuition (This Is the Gold Part)

- **Momentum** tells Adam _where to go_
- **RMSProp** tells Adam _how far to go_
- **Bias correction** makes early learning stable

### 🔑 Rule you should remember

> **Adam = Momentum + Adaptive learning rate + Stability**

---

## 5️⃣ Pure NumPy Code — 3 Hidden Layers + Adam

Below is a **from-scratch Adam implementation**
(no PyTorch, no shortcuts).

---

### 🧠 Network Architecture

```
Input → Dense → ReLU → Dense → ReLU → Dense → ReLU → Output → MSE Loss
```

---

```python
import numpy as np

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# Data
# -----------------------------
N = 64
D = 10
H1, H2, H3 = 32, 16, 8
O = 1

X = np.random.randn(N, D)
y = np.random.randn(N, O)

# -----------------------------
# Hyperparameters
# -----------------------------
lr = 0.001
epochs = 500
eps = 1e-8
beta1 = 0.9
beta2 = 0.999

# -----------------------------
# Initialize parameters
# -----------------------------
def init_w(shape):
    return np.random.randn(*shape) * 0.01

W1, b1 = init_w((D, H1)), np.zeros((1, H1))
W2, b2 = init_w((H1, H2)), np.zeros((1, H2))
W3, b3 = init_w((H2, H3)), np.zeros((1, H3))
W4, b4 = init_w((H3, O)), np.zeros((1, O))

# -----------------------------
# Adam states
# -----------------------------
params = [W1, b1, W2, b2, W3, b3, W4, b4]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]

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
# Training
# -----------------------------
for t in range(1, epochs + 1):

    # ---- Forward ----
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    a3 = relu(z3)

    y_pred = a3 @ W4 + b4

    loss = np.mean((y_pred - y) ** 2)

    # ---- Backward ----
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

    grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]

    # ---- Adam update ----
    for i in range(len(params)):
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)

        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    if t % 50 == 0:
        print(f"Epoch {t} | Loss: {loss:.6f}")

print("\nFinal Loss:", loss)
```

---

## 6️⃣ Which Optimizer Is Better?

### 🔥 Clear Comparison Table

| Optimizer | Strength                 | Weakness               | Verdict             |
| --------- | ------------------------ | ---------------------- | ------------------- |
| Momentum  | Fast convergence         | Same LR for all params | 👍 Good             |
| AdaGrad   | Sparse data              | LR shrinks too much    | ⚠️ Limited          |
| RMSProp   | Fixes AdaGrad            | No momentum            | 👍 Better           |
| **Adam**  | Fast + adaptive + stable | Slightly more compute  | 🏆 **Best default** |

---

## 🏆 Final Verdict (Industry Truth)

> **Adam is the best general-purpose optimizer.**

### Use:

- ✅ **Adam** → almost always (default choice)
- ✅ **RMSProp** → RNNs / unstable gradients
- ⚠️ **AdaGrad** → sparse features only
- ⚠️ **Momentum** → simple problems

---

## 🧠 One-line memory hook

> **Momentum moves fast, AdaGrad adapts, RMSProp stabilizes — Adam does all three.**
