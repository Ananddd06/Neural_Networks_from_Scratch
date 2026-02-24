# 📉 Gradient, Derivatives, and Weight–Bias Update (From Scratch)

---

## 🧠 Goal of Training a Neural Network

The goal of training is to:

> 🎯 **Minimize the loss (error) by updating weights and biases**

This is done using:

- **Derivatives**
- **Gradients**
- **Chain Rule**
- **Gradient Descent**

---

## 🔗 Forward Pass (Basic Formula)

For a single neuron:

### 1️⃣ Linear Combination

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

````

Or in vector form:

```text
z = wᵀx + b
```

Where:

- `w` → weights
- `x` → inputs
- `b` → bias

---

### 2️⃣ Activation Function

```text
a = f(z)
```

Example (Sigmoid):

```text
a = 1 / (1 + e⁻ᶻ)
```

---

## 📊 Loss Function (Error Measure)

### Mean Squared Error (MSE)

```text
L = ½ (y − ŷ)²
```

Where:

- `y` → true label
- `ŷ` → predicted output

📌 The **½** simplifies derivatives.

---

## ❓ Why Derivatives?

Derivatives answer one key question:

> ❓ _If I slightly change a weight or bias, how much does the loss change?_

That’s how learning happens 🧠

---

## 🔁 Backpropagation (Chain Rule)

To update weights and bias, we compute:

```text
∂L / ∂w   and   ∂L / ∂b
```

Using **chain rule**:

```text
∂L/∂w = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂w)
```

```text
∂L/∂b = (∂L/∂ŷ) · (∂ŷ/∂z) · (∂z/∂b)
```

---

## 🧮 Step-by-Step Derivatives (Single Neuron)

### Step 1️⃣ Derivative of Loss w.r.t Output

```text
∂L / ∂ŷ = ŷ − y
```

---

### Step 2️⃣ Derivative of Activation (Sigmoid)

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

Derivative:

```text
∂ŷ / ∂z = σ(z)(1 − σ(z)) = ŷ(1 − ŷ)
```

---

### Step 3️⃣ Derivative of z w.r.t Weight

```text
z = wx + b
```

```text
∂z / ∂w = x
```

```text
∂z / ∂b = 1
```

---

## 🔥 Final Gradient Formulas

### Gradient of Weight

```text
∂L / ∂w = (ŷ − y) · ŷ(1 − ŷ) · x
```

---

### Gradient of Bias

```text
∂L / ∂b = (ŷ − y) · ŷ(1 − ŷ)
```

📌 Bias has no input term → derivative is simpler.

---

## ⬇️ Gradient Descent Update Rule

### Weight Update

```text
w_new = w_old − η · ∂L/∂w
```

### Bias Update

```text
b_new = b_old − η · ∂L/∂b
```

Where:

- `η` (eta) → **learning rate**

---

## 🧠 Intuition Behind the Update

- Gradient tells **direction of maximum error**
- We subtract gradient → move toward **minimum loss**
- Learning rate controls **step size**

```text
Big gradient → big correction
Small gradient → fine tuning
```

---

## 📦 For Multiple Inputs (Summation Form)

For neuron with multiple inputs:

```text
z = Σ (wᵢxᵢ) + b
```

Gradient of each weight:

```text
∂L / ∂wᵢ = (ŷ − y) · f′(z) · xᵢ
```

Bias gradient:

```text
∂L / ∂b = (ŷ − y) · f′(z)
```

📌 Each weight updates **independently**.

---

## 🔄 Why Bias Is Important Here?

- Bias shifts activation left/right
- Helps model learn patterns not passing through origin
- Bias is updated even when input is zero

---

## 🧪 Mini Example (Conceptual)

If:

- Prediction is too high → gradient is positive → weights decrease
- Prediction is too low → gradient is negative → weights increase

👉 This **push–pull** continues until loss is minimum 🎯

---

## 📝 Interview / Exam One-Liners

🔹 **What is gradient?**
👉 Gradient is the rate of change of loss with respect to parameters.

🔹 **Why do we subtract gradient?**
👉 To move in the direction of minimum loss.

🔹 **Why learning rate is important?**
👉 It controls how fast or slow the model learns.

---

## ⭐ Final Key Takeaway

🔥 Loss tells **how wrong**
🔥 Gradient tells **how to correct**
🔥 Weight & bias updates **make learning possible**

> **Training = Repeatedly adjusting weights and bias to minimize loss**

---
