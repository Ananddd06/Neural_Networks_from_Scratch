# 🔗 Chain Rule — The Engine of Backpropagation

## 1. Motivation

Backpropagation uses the **chain rule** from calculus to compute how changing a parameter (weight or bias) changes the overall loss. The chain rule converts a derivative of a composition of functions into a product of derivatives of those functions — allowing gradients to be computed **layer by layer** and reused efficiently.

---

## 2. Scalar chain rule — single nested function

If
$$
z = f(g(x))
$$
then the derivative of \(z\) with respect to \(x\) is:

$$
\frac{dz}{dx} = f'(g(x)) \cdot g'(x)
$$

**Intuition:** a small change \(dx\) causes a change \(dg = g'(x)\,dx\) in the inner function; the outer function \(f\) then changes by \(f'(g(x))\,dg\). Multiply them to get \(dz\).

---

## 3. Worked scalar example

Let
$$
g(x) = x^2,\qquad f(u) = \sin(u).
$$

Then
$$
z = f(g(x)) = \sin(x^2).
$$

Compute derivatives:

$$
g'(x) = 2x,\qquad f'(u) = \cos(u).
$$

Apply chain rule:

$$
\frac{dz}{dx} = f'(g(x)) \cdot g'(x) = \cos(x^2) \cdot 2x.
$$

---

## 4. Nested chain rule — four nested functions

Consider

$$
z = f\big(h\big(m\big(g(x)\big)\big)\big).
$$

Define intermediate variables to make the chain explicit:

$$
u = g(x),\quad v = m(u),\quad w = h(v),\quad z = f(w).
$$

Then by repeated application of the chain rule:

$$
\frac{dz}{dx} = \frac{dz}{dw}\cdot\frac{dw}{dv}\cdot\frac{dv}{du}\cdot\frac{du}{dx}
$$

or written with function derivatives:

$$
\frac{dz}{dx} = f'(w)\cdot h'(v)\cdot m'(u)\cdot g'(x),
$$

where we substitute back:

$$
w = h(m(g(x))),\quad v = m(g(x)),\quad u = g(x).
$$

**Interpretation:** the derivative is the product of the local slopes at each intermediate stage.

---

## 5. Example with four nested functions (simple scalar functions)

Let:

- \(g(x) = 3x\) so \(g'(x)=3\).
- \(m(u) = u^2\) so \(m'(u)=2u\).
- \(h(v) = \log(v)\) so \(h'(v)=\dfrac{1}{v}\) (assume \(v>0\)).
- \(f(w) = e^w\) so \(f'(w)=e^w\).

Then

$$
z = e^{\log\big((3x)^2\big)} = e^{\log(9x^2)} = 9x^2
$$

Direct derivative:

$$
\frac{dz}{dx} = 18x.
$$

Using chain rule:

$$
\frac{dz}{dx} = f'(w)\cdot h'(v)\cdot m'(u)\cdot g'(x)
= e^{w}\cdot \frac{1}{v}\cdot 2u\cdot 3.
$$

Substitute \(u = g(x) = 3x\), \(v = m(u) = (3x)^2 = 9x^2\), \(w = h(v) = \log(9x^2)\):

$$
\frac{dz}{dx} = e^{\log(9x^2)}\cdot\frac{1}{9x^2}\cdot 2(3x)\cdot 3
= 9x^2\cdot\frac{1}{9x^2}\cdot 6x \cdot 3
= 18x
$$

which matches the direct derivative.

---

## 6. Chain rule with parameters (how it applies to backprop)

Neural networks are compositions of **parameterized functions**. Each layer typically computes:

$$
u = W x + b,\quad a = \phi(u)
$$

where \(W\) and \(b\) are parameters to learn and \(\phi\) is an activation.

Suppose the network is

$$
z = f\big(h\big(m\big(g(x; \theta_g);\theta_m\big);\theta_h\big);\theta_f\big),
$$

where \(\theta_g,\theta_m,\theta_h,\theta_f\) are parameters in the corresponding modules (weights and biases). We want derivatives of the loss \(L(z,y)\) with respect to each parameter.

Define the loss \( \mathcal{L} = L(z,y) \). By chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \theta_g}
=
\frac{\partial \mathcal{L}}{\partial z}\cdot
\frac{\partial z}{\partial h}\cdot
\frac{\partial h}{\partial m}\cdot
\frac{\partial m}{\partial g}\cdot
\frac{\partial g}{\partial \theta_g}.
$$

Analogous formulas hold for \( \theta_m, \theta_h, \theta_f \), where the chain product stops at the parameter's location.

**Key idea:** each gradient is a product of local derivatives (scalars, vectors or Jacobians) from the output back to the parameter.

---

## 7. Matrix/vector (multivariable) chain rule and Jacobians

When quantities are vectors, we use Jacobians. If \(y=\sigma(Ax)\) with \(A\in\mathbb{R}^{m\times n}\), then:

- The derivative of \(Ax\) w.r.t. \(x\) is the matrix \(A\).
- If \(\sigma\) is elementwise, its Jacobian is a diagonal matrix with \(\sigma'\) on the diagonal.

For a composition \(z = f(g(x))\) with \(g:\mathbb{R}^n\to\mathbb{R}^m\) and \(f:\mathbb{R}^m\to\mathbb{R}\), the gradient is:

$$
\nabla_x z = J_g(x)^\top \nabla_u f(u)\big|_{u=g(x)}
$$

where \(J_g(x)\) is the Jacobian matrix of \(g\) at \(x\), and \(\nabla_u f\) is the gradient of \(f\) w.r.t. its vector input.

In neural nets the repeated application of this rule yields efficient matrix forms used in backprop.

---

## 8. Concrete neural-network style example with 4 layers

Consider a simple 4-layer feedforward chain (scalar shapes shown for clarity; vectorization is analogous):

Layer definitions:

- Layer 1 (closest to input):
  $$
  u_1 = g(x) = W_1 x + b_1,\qquad a_1 = \phi_1(u_1)
  $$

- Layer 2:
  $$
  u_2 = m(a_1) = W_2 a_1 + b_2,\qquad a_2 = \phi_2(u_2)
  $$

- Layer 3:
  $$
  u_3 = h(a_2) = W_3 a_2 + b_3,\qquad a_3 = \phi_3(u_3)
  $$

- Layer 4 (output pre-activation):
  $$
  u_4 = f(a_3) = W_4 a_3 + b_4,\qquad z = \phi_4(u_4)
  $$

Loss:
$$
\mathcal{L} = L(z, y)
$$

We want gradients w.r.t. parameters \(W_i, b_i\).

### Step A — compute output error term

Start from the loss:

$$
\frac{\partial \mathcal{L}}{\partial u_4}
= \frac{\partial \mathcal{L}}{\partial z}\cdot \phi_4'(u_4).
$$

Define the *delta* at layer 4:

$$
\delta_4 \equiv \frac{\partial \mathcal{L}}{\partial u_4}.
$$

### Step B — gradients for \(W_4, b_4\)

Using the local rule (linear transform):

$$
\frac{\partial \mathcal{L}}{\partial W_4} = \delta_4 \cdot a_3^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b_4} = \delta_4.
$$

### Step C — propagate the delta backward

The delta for layer 3:

$$
\delta_3 = (W_4^\top \delta_4) \odot \phi_3'(u_3),
$$

where \(\odot\) is elementwise (Hadamard) product when activations are elementwise.

Then

$$
\frac{\partial \mathcal{L}}{\partial W_3} = \delta_3 \cdot a_2^\top,
\qquad
\frac{\partial \mathcal{L}}{\partial b_3} = \delta_3.
$$

### Step D — repeat back to earlier layers

Similarly,

$$
\delta_2 = (W_3^\top \delta_3) \odot \phi_2'(u_2),
\qquad
\frac{\partial \mathcal{L}}{\partial W_2} = \delta_2 \cdot a_1^\top.
$$

And

$$
\delta_1 = (W_2^\top \delta_2) \odot \phi_1'(u_1),
\qquad
\frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 \cdot x^\top.
$$

**This is precisely the repeated application of the chain rule.**

---

## 9. Parameter update rules (gradient descent)

Once we have the gradients, update each parameter (here using simple gradient descent):

$$
W_i \leftarrow W_i - \eta \frac{\partial \mathcal{L}}{\partial W_i},
\qquad
b_i \leftarrow b_i - \eta \frac{\partial \mathcal{L}}{\partial b_i}.
$$

The required derivatives \( \frac{\partial \mathcal{L}}{\partial W_i} \) are computed using the chain rule products summarized above.

---

## 10. Small numerical illustration (one neuron per layer, scalar)

Let input \(x=2\) and consider:

- \(W_1 = 1,\, b_1 = 0,\; \phi_1(u)=u\) (identity)
- \(W_2 = 2,\, b_2 = 0,\; \phi_2(u)=u\)
- \(W_3 = 1,\, b_3 = 0,\; \phi_3(u)=\tanh(u)\)
- \(W_4 = 1,\, b_4 = 0,\; \phi_4(u)=u\) (identity)
- Loss \( \mathcal{L} = \tfrac{1}{2}(z - y)^2 \) with target \(y=10\)

Forward pass:

1. \(u_1 = W_1 x + b_1 = 1\cdot 2 + 0 = 2,\; a_1 = 2.\)
2. \(u_2 = W_2 a_1 + b_2 = 2\cdot 2 = 4,\; a_2 = 4.\)
3. \(u_3 = W_3 a_2 + b_3 = 1\cdot 4 = 4,\; a_3 = \tanh(4)\approx 0.9993.\)
4. \(u_4 = W_4 a_3 + b_4 = 0.9993,\; z = 0.9993.\)
5. Loss: \( \mathcal{L} = \tfrac{1}{2}(0.9993-10)^2 \approx 40.02.\)

Backward pass:

- Output delta:
  $$
  \frac{\partial \mathcal{L}}{\partial z} = z - y \approx 0.9993 - 10 = -9.0007.
  $$
  Since \(\phi_4\) is identity,
  $$
  \delta_4 = -9.0007.
  $$

- Gradients for \(W_4, b_4\):
  $$
  \frac{\partial \mathcal{L}}{\partial W_4} = \delta_4 \cdot a_3 \approx -9.0007 \cdot 0.9993 \approx -8.994.
  $$
  $$
  \frac{\partial \mathcal{L}}{\partial b_4} = \delta_4 \approx -9.0007.
  $$

- Propagate to layer 3:
  $$
  \delta_3 = W_4^\top \delta_4 \odot \phi_3'(u_3).
  $$
  Here \(\phi_3(u)=\tanh(u)\) so \(\phi_3'(u)=1-\tanh^2(u)\).
  Numerically \(\phi_3'(4)=1-(0.9993)^2\approx 0.0014\).
  $$
  \delta_3 \approx 1\cdot(-9.0007)\cdot 0.0014 \approx -0.0126.
  $$

- Gradients for \(W_3, b_3\):
  $$
  \frac{\partial \mathcal{L}}{\partial W_3} = \delta_3 \cdot a_2 \approx -0.0126 \cdot 4 \approx -0.0504.
  $$
  $$
  \frac{\partial \mathcal{L}}{\partial b_3} = \delta_3 \approx -0.0126.
  $$

- Continue similarly for \(W_2, b_2\) and \(W_1, b_1\). Each step multiplies by the upstream weight and the local derivative (this is the chain rule in practice).

If learning rate \(\eta = 0.1\), update \(W_4\):

$$
W_4 \leftarrow W_4 - 0.1\cdot(-8.994) \approx 1 + 0.8994 \approx 1.8994.
$$

This numeric example shows how local derivatives combine multiplicatively to provide each parameter's gradient.

---

## 11. Summary & takeaways

- The **chain rule** turns derivatives of nested functions into a product of **local derivatives**.
- In backpropagation we compute **local derivatives (activations' derivatives and linear derivatives)** and multiply them moving backward — reusing computed intermediate results.
- For vector-valued layers we use **Jacobians** and the matrix form:
  $$
  \delta^{[l]} = (W^{[l+1]})^\top \delta^{[l+1]} \odot \phi'(u^{[l]}).
  $$
- Gradients for weights are local:
  $$
  \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]}\cdot (a^{[l-1]})^\top,\qquad
  \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}.
  $$
- Once gradients are computed, parameters are updated using gradient descent.

---

## References (for further reading)

- Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning* (book)
- CS231n (Stanford) lecture notes — backpropagation & chain rule

---

**End of document.**
