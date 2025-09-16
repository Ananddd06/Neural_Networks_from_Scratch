# 🧠 Neural Network Basics

## 1. What is a Neural Network?

A **Neural Network (NN)** is a **mathematical model** inspired by how the human brain processes information.

At its core:

- It takes **inputs** (features of your data) 📊
- Processes them through **layers of artificial neurons** 🔗
- Produces an **output** (like a prediction) 🎯

Each neuron is basically a **tiny function** that transforms input into something useful for solving the overall problem (classification, regression, translation, etc.).

---

## 2. The Formula: `x1w1 + x2w2 + b`

This is the **basic operation inside a neuron** ⚡.

### Decode it:

- **Inputs (`x1, x2, …, xn`)** → Features of your data 🏠📐
- **Weights (`w1, w2, …, wn`)** → "Importance multipliers" ⚖️ Learned during training 🤖
- **Bias (`b`)** → Baseline adjustment ➕ Shifts the decision boundary

So, a neuron computes:

```math
z = (x1w1 + x2w2 + … + xnwn) + b

```

Where `z` is the **weighted sum** of inputs plus bias.

---

## 3. Why Not Stop at `z`?

After computing `z`, the neuron applies an **activation function** (like sigmoid, ReLU, tanh).

`a = f(z)`

This adds **non-linearity**, letting neural networks model complex patterns. Without activation, the NN would just be a **linear model** (like linear regression).

---

## 4. What’s the Role of Each Part?

- **Inputs (`xᵢ`)**: The raw information from your data.
- **Weights (`wᵢ`)**: Learn which features matter more.
- **Bias (`b`)**: Gives flexibility to shift the function.
- **Activation `f(z)`**: Makes the neuron capable of modeling non-linear relationships.

---

## 5. Putting it Together (One Neuron Example)

Imagine a neuron for predicting whether a student will **pass an exam**:

**Inputs:**

- `x₁` = hours studied
- `x₂` = hours slept

**Weights:**

- `w₁ = 0.7` (studying matters more)
- `w₂ = 0.3` (sleep matters but less)

**Bias:**

- `b = -5` (baseline difficulty of the exam)

**Calculation:**

`z = (x₁ ⋅ 0.7) + (x₂ ⋅ 0.3) - 5`

Then activation (say sigmoid):

`a = 1 / (1 + e⁻ᶻ)`

This gives a probability between 0 and 1. If `a > 0.5`, the model predicts "Pass".

---

✅ **In short:**

- **Weights & bias** are what the network learns.
- **Input × weight + bias** is how each neuron processes data.
- **Activation** makes the network powerful enough to model any function.

---
