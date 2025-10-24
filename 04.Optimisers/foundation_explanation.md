# Neural Network from Scratch: Forward Pass and Optimization

This repository contains a Jupyter Notebook (`Foundation.ipynb`) that demonstrates the fundamental concepts of a neural network. It walks through building core components from scratch using NumPy and then explores different optimization strategies to illustrate *why* a systematic approach like gradient descent is necessary.

## Overview

The notebook follows a clear pedagogical path:

1.  **Build It (NumPy):** Implement the core components of a neural network (Dense Layer, ReLU, Softmax, and Cross-Entropy Loss) using only NumPy.
2.  **Forward Pass:** Assemble these components into a simple network and perform a single forward pass to see the (untrained) output.
3.  **Compare It (PyTorch):** Build the *exact same* network in PyTorch to serve as a reference and show how a modern framework handles the forward and backward passes.
4.  **Optimize It (The "Wrong" Way):** Attempt to "train" the NumPy network using two different random search strategies.
5.  **The "Why":** Demonstrate why these random optimization methods fail on complex, non-linear data (the "spiral" dataset), proving the necessity for a more intelligent optimization algorithm like gradient descent and backpropagation.

-----

## Core Components (Implemented in NumPy)

The notebook defines the following classes from scratch:

  * `Dense_Layer`: A standard fully-connected layer.
      * Initializes weights with small random values (`0.01 * np.random.randn`) and biases as zeros.
      * Performs the forward pass calculation: $output = inputs \cdot weights + biases$.
  * `Activation_Relu`: The Rectified Linear Unit (ReLU) activation function.
      * Performs the forward pass: $output = \max(0, inputs)$.
  * `Activation_Softmax`: The Softmax activation function, used for the output layer in multi-class classification.
      * Calculates exponential values and normalizes them to produce a probability distribution.
      * Includes a stabilization step (`- np.max(...)`) to prevent overflow.
  * `Loss_CategoricalCrossentropy`: The categorical cross-entropy loss function.
      * Calculates the negative log-likelihood of the correct class predictions.
      * Includes clipping (`np.clip`) to prevent taking the log of zero.

-----

## Experiments and Key Takeaways

The notebook is structured as a series of experiments.

### 1\. NumPy Forward Pass

A 2-layer network is built using the from-scratch classes and fed the `vertical_data` dataset. This single forward pass with uninitialized weights results in an accuracy of \~33% and a loss of \~1.098, which is the expected result for random guessing among three classes ($-\log(1/3)$).

### 2\. PyTorch Comparison

An identical network is created using `torch.nn.Sequential`. This cell serves as a validation of the NumPy implementation and also demonstrates the standard training loop, including `loss.backward()` and `optimizer.step()`, which are the parts (backpropagation and gradient descent) missing from the NumPy model.

### 3\. Optimization Strategy 1: Full Random Search

This experiment attempts to find the best network parameters by randomly generating *completely new* sets of weights and biases for 10,000 iterations. It keeps track of the set that produces the lowest loss.

  * **Result:** This method is very inefficient but manages to find a decent set of parameters for the simple `vertical_data`, achieving an accuracy of \~80-90%.

### 4\. Optimization Strategy 2: Random Adjustment (Hill Climbing)

This experiment tries a slightly "smarter" approach. Instead of generating totally new weights, it starts with one set and, in each iteration, *adds small random values* to them. If this "tweak" improves the loss, the new weights are kept. If not, they are reverted.

  * **Result on `vertical_data`:** This method performs surprisingly well on the simple, linearly separable `vertical_data`. It quickly finds a combination of weights that achieves high accuracy (\>90%).

### 5\. The Failure Case: Why We Need Gradient Descent

The success of the random adjustment method in Experiment 4 begs the question: "If this works, why do we need complicated backpropagation and gradient descent?"

This final experiment answers that question.

  * **Action:** The *exact same code* from Experiment 4 (random adjustment) is run on the much more complex, non-linear `spiral_data` dataset.
  * **Result on `spiral_data`:** The method **fails completely**. The accuracy remains stuck at \~33% (random guessing), and the loss never improves.

## Conclusion

This notebook perfectly illustrates that while a "blind" random search for weights can work on trivial problems, it stands no chance against complex, non-linear data. The search space is simply too vast to find a good solution by chance.

This failure motivates the need for a more intelligent optimization strategy: **Gradient Descent**. Instead of guessing randomly, gradient descent calculates the *gradient* (the derivative) of the loss function with respect to each weight and bias. This gradient tells the optimizer the *exact direction* to "tweak" the parameters to most efficiently decrease the loss. The algorithm used to calculate these gradients efficiently is **Backpropagation**.

-----

## How to Run

1.  Ensure you have Python and Jupyter Notebook installed.
2.  Install the required libraries:
    ```bash
    pip install nnfs numpy pandas torch
    ```
3.  Launch Jupyter Notebook and open `Foundation.ipynb`.
4.  Run the cells in order to follow the narrative.
