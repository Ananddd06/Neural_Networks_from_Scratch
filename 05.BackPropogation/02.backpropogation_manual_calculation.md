# 🔁 Backpropagation from Scratch (Sigmoid Activation)

## Full Derivation + Chain Rule + Manual Calculation (2 Inputs)

---

## 🎯 Goal of Backpropagation

> **Minimize the loss by updating weights and biases**

Backpropagation tells us:

- ❓ _How much each weight & bias contributes to the error_
- 🔁 _How to correct them using gradients_

---

## 🧠 Network Architecture (Simple & Clear)

We use:

- **2 input neurons**
- **1 hidden neuron**
- **1 output neuron**
- **Sigmoid activation everywhere**
- **Mean Squared Error loss**

```

x1 ---- w1 ----

> (Hidden neuron) ---- w3 ----
> x2 ---- w2 ----/ > (Output)

- b1 + b2

```

---

## 🧪 Step 0: Given Sample Values

### Inputs

```

x₁ = 0.5
x₂ = 0.8

```

### Weights & Biases (Random Initialization)

```

w₁ = 0.4
w₂ = -0.2
b₁ = 0.1

w₃ = 0.7
b₂ = -0.3

```

### Target Output

```

y = 1

```

### Learning Rate

```

η = 0.1

```

---

## 🔗 Step 1: Forward Propagation

---

### 🟢 Hidden Layer

#### Linear Combination

```

z₁ = w₁x₁ + w₂x₂ + b₁

```

Substitute values:

```

z₁ = (0.4 × 0.5) + (-0.2 × 0.8) + 0.1
z₁ = 0.2 - 0.16 + 0.1
z₁ = 0.14

```

#### Sigmoid Activation

```

a₁ = σ(z₁) = 1 / (1 + e⁻ᶻ¹)

```

```

a₁ = σ(0.14) ≈ 0.535

```

---

### 🔵 Output Layer

#### Linear Combination

```

z₂ = w₃a₁ + b₂

```

```

z₂ = (0.7 × 0.535) + (-0.3)
z₂ = 0.3745 - 0.3
z₂ = 0.0745

```

#### Sigmoid Activation (Prediction)

```

ŷ = σ(z₂)

```

```

ŷ = σ(0.0745) ≈ 0.5186

```

---

## 📉 Step 2: Loss Calculation (MSE)

```

L = ½ (y − ŷ)²

```

```

L = ½ (1 − 0.5186)²
L = ½ (0.4814)²
L ≈ 0.116

```

---

## 🔁 Step 3: Backpropagation (CHAIN RULE)

Backpropagation = **chain rule applied repeatedly**

---

## 🔴 Output Layer Gradients

### 1️⃣ Derivative of Loss w.r.t Output

```

∂L / ∂ŷ = ŷ − y

```

```

∂L / ∂ŷ = 0.5186 − 1 = −0.4814

```

---

### 2️⃣ Derivative of Sigmoid

```

σ′(z) = σ(z)(1 − σ(z))

```

```

∂ŷ / ∂z₂ = 0.5186(1 − 0.5186)
∂ŷ / ∂z₂ ≈ 0.2496

```

---

### 3️⃣ Chain Rule (Output Layer)

```

∂L / ∂z₂ = (∂L/∂ŷ)(∂ŷ/∂z₂)

```

```

∂L / ∂z₂ = (−0.4814)(0.2496)
∂L / ∂z₂ ≈ −0.120

```

---

### 4️⃣ Gradients of w₃ and b₂

```

∂z₂ / ∂w₃ = a₁
∂z₂ / ∂b₂ = 1

```

```

∂L / ∂w₃ = (−0.120)(0.535) ≈ −0.064
∂L / ∂b₂ = −0.120

```

---

## 🔵 Hidden Layer Gradients (Chain Rule Continues)

### Why More Chain Rule?

Because hidden neurons **don’t directly see the loss**.

---

### 5️⃣ Backprop Error to Hidden Layer

```

∂L / ∂a₁ = (∂L / ∂z₂)(∂z₂ / ∂a₁)

```

```

∂L / ∂a₁ = (−0.120)(w₃)
∂L / ∂a₁ = (−0.120)(0.7)
∂L / ∂a₁ = −0.084

```

---

### 6️⃣ Sigmoid Derivative (Hidden Layer)

```

∂a₁ / ∂z₁ = a₁(1 − a₁)

```

```

∂a₁ / ∂z₁ = 0.535(1 − 0.535)
∂a₁ / ∂z₁ ≈ 0.249

```

---

### 7️⃣ Hidden Layer Delta

```

∂L / ∂z₁ = (∂L/∂a₁)(∂a₁/∂z₁)

```

```

∂L / ∂z₁ = (−0.084)(0.249)
∂L / ∂z₁ ≈ −0.0209

```

---

### 8️⃣ Gradients of w₁, w₂, b₁

```

∂z₁ / ∂w₁ = x₁
∂z₁ / ∂w₂ = x₂
∂z₁ / ∂b₁ = 1

```

```

∂L / ∂w₁ = (−0.0209)(0.5) = −0.01045
∂L / ∂w₂ = (−0.0209)(0.8) = −0.0167
∂L / ∂b₁ = −0.0209

```

---

## ⬇️ Step 4: Weight & Bias Updates

### Output Layer

```

w₃_new = 0.7 − 0.1(−0.064) = 0.7064
b₂_new = −0.3 − 0.1(−0.120) = −0.288

```

### Hidden Layer

```

w₁_new = 0.4 − 0.1(−0.01045) = 0.4010
w₂_new = −0.2 − 0.1(−0.0167) = −0.1983
b₁_new = 0.1 − 0.1(−0.0209) = 0.1021

```

---

## 📊 Summary Table

| Parameter | Old  | New         |
| --------- | ---- | ----------- |
| w₁        | 0.4  | **0.4010**  |
| w₂        | -0.2 | **-0.1983** |
| b₁        | 0.1  | **0.1021**  |
| w₃        | 0.7  | **0.7064**  |
| b₂        | -0.3 | **-0.288**  |

📌 **Loss decreased → learning happened 🎯**

---

## 🧠 Key Intuition (VERY IMPORTANT)

- 🔥 **Forward pass** → prediction
- 🔥 **Loss** → how wrong
- 🔥 **Chain rule** → blame assignment
- 🔥 **Gradients** → correction signal
- 🔥 **Gradient descent** → learning

---

## ⭐ Final One-Line Definition (Exam Gold)

> **Backpropagation is an algorithm that applies the chain rule to compute gradients of the loss with respect to weights and biases, enabling a neural network to learn.**

---
