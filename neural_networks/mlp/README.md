# Multilayer Perceptron (MLP) for Financial Forecasting

A Multilayer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). It is the most classic type of neural network, composed of an input layer, one or more hidden layers of neurons, and an output layer. MLPs are capable of learning complex, non-linear relationships in data, making them suitable for a wide range of financial forecasting tasks.

## Mathematical Formulation

An MLP consists of multiple layers of nodes, where each node is a neuron (or perceptron) that uses a non-linear activation function. The network learns through a process of forward propagation and backpropagation.

### 1. Forward Propagation

For a single neuron in a layer, the output is calculated by taking a weighted sum of the inputs from the previous layer, adding a bias, and then passing the result through an activation function.

The output \( a_j \) of a neuron \( j \) is:

\[ a_j = g(z_j) = g \left( \sum_{i=1}^{n} w_{ij} x_i + b_j \right) \]

Where:
-   \( x_i \) are the inputs from the previous layer.
-   \( w_{ij} \) are the weights connecting input \( i \) to neuron \( j \).
-   \( b_j \) is the bias term for neuron \( j \).
-   \( g(z) \) is the activation function.

Common activation functions include:
-   **ReLU (Rectified Linear Unit):** \( g(z) = \max(0, z) \). This is the most common choice for hidden layers.
-   **Sigmoid:** \( g(z) = \frac{1}{1 + e^{-z}} \). Used for binary classification output layers.
-   **Softmax:** Used for multi-class classification output layers.
-   **Linear:** Used for regression output layers.

This process is repeated layer by layer, from the input layer to the output layer, to generate a final prediction.

### 2. Loss Function and Backpropagation

After making a prediction, the network calculates the error using a **loss function** (e.g., Mean Squared Error for regression, Cross-Entropy for classification). The goal of training is to adjust the weights and biases to minimize this loss.

This is achieved via **backpropagation**. The algorithm calculates the gradient of the loss function with respect to each weight and bias in the network, starting from the output layer and moving backward. An **optimizer** (like Adam, RMSprop, or SGD) then uses these gradients to update the weights and biases in the direction that minimizes the loss.

\[ w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w} \]

Where \( \eta \) is the learning rate, a hyperparameter that controls the step size of the weight updates.

## Application in Finance

-   **Time Series Forecasting:** Predict future stock prices or returns based on a window of past data.
-   **Credit Scoring:** Model complex, non-linear patterns in a customer's financial history to predict default risk.
-   **Algorithmic Trading:** Identify complex trading signals from a variety of market inputs.
