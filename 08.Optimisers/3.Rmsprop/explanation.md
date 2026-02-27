# 📘 RMSProp Optimizer — Fresher-Friendly, Detailed Explanation

---

## 1️⃣ Why RMSProp Exists (The Problem It Solves)

### Problem with AdaGrad (quick recap)

AdaGrad updates like this:

$$
w = w - \frac{\eta}{\sqrt{\sum g^2} + \epsilon} \cdot g
$$

❌ The sum $\sum g^2$ **keeps growing forever**
❌ Learning rate becomes **almost zero**
❌ Training stops early

👉 **RMSProp fixes this**

---

## 2️⃣ Big Idea Behind RMSProp (Plain English)

> **Don’t remember all past gradients — remember only recent ones.**

Instead of **summing** squared gradients forever,
RMSProp keeps a **moving average**.

So:

- Old gradients slowly fade away
- Learning rate never becomes zero

---

## 3️⃣ What RMSProp Stores (Very Simple)

For **each weight & bias**, RMSProp stores:

🧠 **One memory**:

- Moving average of **squared gradients**

This is why RMSProp is **simpler than Adam**.

---

## 4️⃣ RMSProp Step-by-Step Math (Beginner Level)

Let current gradient be:

$$
g_t = \frac{\partial L}{\partial w}
$$

---

### 🔹 Step 1: Moving average of squared gradients

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t^2
$$

Where:

- $\beta$ is decay rate (usually **0.9**)

**Meaning:**

- 90% past squared gradients
- 10% current squared gradient

---

### 🔹 Step 2: Weight update

$$
w = w - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t
$$

**Interpretation:**

- Large gradients → big $v_t$ → smaller step
- Small gradients → small $v_t$ → larger step

---

## 5️⃣ Key Fresher Intuition (VERY IMPORTANT)

> **RMSProp adjusts learning rate automatically using recent gradients only.**

Think like this:

- “If gradients are unstable → slow down”
- “If gradients are small → speed up”

---

## 6️⃣ RMSProp vs AdaGrad (Freshers View)

| Feature                | AdaGrad         | RMSProp             |
| ---------------------- | --------------- | ------------------- |
| Gradient memory        | All history     | Recent history only |
| Learning rate          | Keeps shrinking | Stays healthy       |
| Suitable for deep nets | ❌ No           | ✅ Yes              |

---

## 7️⃣ RMSProp vs Adam (Simple Words)

| Feature         | RMSProp | Adam             |
| --------------- | ------- | ---------------- |
| Momentum        | ❌ No   | ✅ Yes           |
| Adaptive LR     | ✅ Yes  | ✅ Yes           |
| Bias correction | ❌ No   | ✅ Yes           |
| Complexity      | Simple  | Slightly complex |

👉 **Adam = RMSProp + Momentum**

---

## 8️⃣ RMSProp Algorithm (Beginner Pseudocode)

```
initialize weights W, bias b
initialize v = 0

for each iteration:
    compute gradient g
    v = beta * v + (1 - beta) * g^2
    W = W - lr * g / (sqrt(v) + eps)
```

---

## 9️⃣ Pure NumPy Code — 3 Hidden Layers + RMSProp

This is **fully from scratch**, beginner-readable.

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
beta = 0.9
eps = 1e-8

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
# RMSProp states
# -----------------------------
vW1, vb1 = np.zeros_like(W1), np.zeros_like(b1)
vW2, vb2 = np.zeros_like(W2), np.zeros_like(b2)
vW3, vb3 = np.zeros_like(W3), np.zeros_like(b3)
vW4, vb4 = np.zeros_like(W4), np.zeros_like(b4)

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

    # ---- RMSProp update ----
    vW4 = beta * vW4 + (1 - beta) * (dW4 ** 2)
    vb4 = beta * vb4 + (1 - beta) * (db4 ** 2)
    W4 -= lr * dW4 / (np.sqrt(vW4) + eps)
    b4 -= lr * db4 / (np.sqrt(vb4) + eps)

    vW3 = beta * vW3 + (1 - beta) * (dW3 ** 2)
    vb3 = beta * vb3 + (1 - beta) * (db3 ** 2)
    W3 -= lr * dW3 / (np.sqrt(vW3) + eps)
    b3 -= lr * db3 / (np.sqrt(vb3) + eps)

    vW2 = beta * vW2 + (1 - beta) * (dW2 ** 2)
    vb2 = beta * vb2 + (1 - beta) * (db2 ** 2)
    W2 -= lr * dW2 / (np.sqrt(vW2) + eps)
    b2 -= lr * db2 / (np.sqrt(vb2) + eps)

    vW1 = beta * vW1 + (1 - beta) * (dW1 ** 2)
    vb1 = beta * vb1 + (1 - beta) * (db1 ** 2)
    W1 -= lr * dW1 / (np.sqrt(vW1) + eps)
    b1 -= lr * db1 / (np.sqrt(vb1) + eps)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

print("\nFinal Loss:", loss)
```

---

## 🔟 Final Fresher Summary (Memorize This)

> 🔹 **AdaGrad** remembers everything → slows too much
> 🔹 **RMSProp** remembers recent past → stable learning
> 🔹 **Adam** remembers direction + speed → best default

### One-line rule:

> **RMSProp = AdaGrad with forgetting**

---
