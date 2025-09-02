# Neural Networks from Scratch

This repository is a structured guide to understanding and building **Neural Networks from Scratch**.  
It covers the fundamental concepts, mathematics, and implementation details required to design, train, and evaluate neural networks without relying on high-level libraries.

---

## 📘 Table of Contents

1. Introduction to Neural Networks
2. Biological Inspiration
3. Neurons & Perceptrons
4. Activation Functions
5. Forward Propagation
6. Loss Functions
7. Gradient Descent & Backpropagation
8. Optimization Algorithms
9. Regularization Techniques
10. Deep Neural Networks
11. Convolutional Neural Networks (CNNs)
12. Recurrent Neural Networks (RNNs)
13. Training Best Practices
14. Projects & Applications
15. References & Further Reading

---

## 1. Introduction to Neural Networks

- What is a Neural Network?
- Historical background (Perceptron → Deep Learning)
- Why build from scratch?

## 2. Biological Inspiration

- Neurons in the human brain
- How artificial neurons are modeled
- Synaptic weights and signal flow

## 3. Neurons & Perceptrons

- Structure of a perceptron
- Weighted sum and bias
- Linear separability
- XOR problem and limitations

## 4. Activation Functions

- **Sigmoid** → Probability interpretation
- **Tanh** → Centered outputs
- **ReLU** → Sparse activations & efficiency
- **Leaky ReLU, ELU, GELU**
- When to use which activation

## 5. Forward Propagation

- Layer-wise computation
- Matrix representation
- Batch inputs
- Example walkthrough with small NN

## 6. Loss Functions

- **Regression**: Mean Squared Error (MSE)
- **Classification**: Cross-Entropy Loss
- Hinge Loss, KL Divergence
- Intuition behind minimizing loss

## 7. Gradient Descent & Backpropagation

- Intuition of gradients
- Derivatives of activation functions
- Backpropagation algorithm step-by-step
- Chain rule in action
- Vanishing/Exploding gradients

## 8. Optimization Algorithms

- **Batch Gradient Descent**
- **Stochastic Gradient Descent (SGD)**
- Momentum
- RMSProp
- Adam Optimizer
- Learning rate schedules

## 9. Regularization Techniques

- Overfitting vs Underfitting
- **L1 / L2 Regularization**
- **Dropout**
- **Batch Normalization**
- Early Stopping
- Data Augmentation

## 10. Deep Neural Networks

- Stacking multiple layers
- Universal Approximation Theorem
- Initialization strategies (Xavier, He)
- Depth vs Width trade-offs

## 11. Convolutional Neural Networks (CNNs)

- Convolution operation
- Filters & Feature maps
- Pooling layers
- CNN architectures (LeNet, AlexNet, ResNet)

## 12. Recurrent Neural Networks (RNNs)

- Sequential data handling
- Vanilla RNNs
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Applications in NLP and time-series

## 13. Training Best Practices

- Data preprocessing & normalization
- Mini-batch training
- Hyperparameter tuning
- Model evaluation metrics (Accuracy, F1, ROC-AUC)
- Debugging training issues

## 14. Projects & Applications

- Handwritten digit recognition (MNIST)
- Image classification
- Sentiment analysis
- Time-series prediction
- Neural style transfer

## 15. References & Further Reading

- **Books**:
  - "Deep Learning" by Ian Goodfellow
  - "Neural Networks and Deep Learning" by Michael Nielsen
- **Courses**:
  - Andrew Ng’s Deep Learning Specialization
  - Stanford CS231n (CNNs for Visual Recognition)
  - MIT 6.S191 (Intro to Deep Learning)
- **Research Papers**:
  - Perceptron (Rosenblatt, 1958)
  - Backpropagation (Rumelhart et al., 1986)
  - Deep Residual Learning (He et al., 2015)

---

## 🚀 Goal

By completing this journey, you will:

- Understand neural networks at the mathematical and conceptual level.
- Implement networks using only **NumPy**.
- Build intuition to debug and optimize deep learning models.
