# Multilayer-Perceptron-from-Scratch
This repository contains a from-scratch implementation of a **Multilayer Perceptron (MLP)** using NumPy to classify handwritten digits from the MNIST dataset. The implementation demonstrates the fundamental building blocks of deep learning without the use of high-level frameworks like PyTorch or TensorFlow.

---

## Overview of the Code

The implementation is encapsulated within the `SimpleNN` class, which manages the lifecycle of the neural network:

* **Initialization**: Uses **He Initialization**, which is specifically designed for layers followed by ReLU activation functions.
* **Forward Propagation**: Computes the linear transformation  and applies activation functions (ReLU for hidden layers, Softmax for the output layer) across the specified architecture.
* **Backward Propagation**: Implements the chain rule to compute gradients for weights and biases, including an **L2 Regularization** term to penalize large weights.
* **Optimization**: Updates parameters using **Stochastic Gradient Descent (SGD)** combined with **Learning Rate Decay** and **Gradient Clipping** to ensure stable training.

---

## Core Topics Covered

### 1. Activation Functions

* **ReLU (Rectified Linear Unit)**: Defined as , it is used in hidden layers to introduce non-linearity and reduce the likelihood of vanishing gradients.
* **Softmax**: Applied at the output layer to convert raw scores into a probability distribution across the 10 digit classes.

### 2. Parameter Initialization

* **He Initialization**: Weights are initialized using a normal distribution scaled by , where  is the number of input units to the layer. This prevents the signal from dying out or exploding during the first few passes.

### 3. Loss & Regularization

* **Cross-Entropy Loss**: Measures the difference between the predicted probability distribution and the actual one-hot encoded labels.
* **L2 Regularization**: Adds  to the loss function to discourage overfitting by penalizing large weight values.

### 4. Training Optimization Techniques

* **Mini-batch Gradient Descent**: The training data is shuffled and split into small batches (e.g., 64) to provide a balance between the efficiency of batch processing and the stochastic nature of individual samples.
* **Learning Rate Decay**: The learning rate is reduced by a decay rate at specific epoch intervals to help the model converge more precisely as training progresses.
* **Gradient Clipping**: Gradients are clipped between  to prevent "exploding gradients" which can lead to numerical instability (NaN values).

---

## Dataset: MNIST

The model is trained on the **MNIST** dataset of handwritten digits:

* **Input**:  pixel images (784 features) normalized to a range of .
* **Output**: 10 classes representing digits 0–9.
* **Splits**: The code utilizes 50,000 samples for training, 10,000 for validation, and 10,000 for testing.

---

## Performance Summary

Based on the notebook execution results:

* **Network Architecture**: `[784, 512, 256, 128, 10]`.
* **Training Specs**: 20 Epochs, 0.01 Initial Learning Rate, Batch Size of 64.
* **Final Accuracy**: The model achieved a **Validation Accuracy of 79.19%** and a **Test Accuracy of 80.24%**.
* **Progress**: The implementation successfully improved accuracy from an initial random-guess baseline of ~12% to over 80%.

---

## Usage

To run this project, ensure you have the following Python libraries:

```bash
pip install numpy matplotlib scipy scikit-learn

```

**Would you like me to show you how to implement a different optimizer, such as Adam, to further improve the accuracy of this model?**
