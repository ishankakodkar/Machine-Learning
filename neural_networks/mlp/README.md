# Multilayer Perceptron (MLP) in Finance

A Multilayer Perceptron (MLP) is the classic type of feedforward artificial neural network. It serves as the foundation for more complex deep learning models. In finance, MLPs are versatile tools that can be applied to a wide range of prediction problems where there are non-linear relationships between inputs and outputs.

## Structure

An MLP consists of:

1.  **Input Layer:** Receives the initial data or features (e.g., historical prices, technical indicators).
2.  **Hidden Layers:** One or more layers of neurons that sit between the input and output layers. These layers perform computations and allow the network to learn complex patterns.
3.  **Output Layer:** Produces the final prediction (e.g., the probability of a stock price increase).

Each neuron in a layer is connected to all neurons in the next layer, and this is why they are often called "fully connected" networks.

## Application in Finance

MLPs can be used for both regression (predicting a continuous value) and classification (predicting a category). Common applications include:

-   **Stock Price Movement Prediction:** Classifying whether a stock will go up or down.
-   **Credit Scoring:** Predicting whether a borrower will default on a loan.
-   **Algorithmic Trading:** Generating buy/sell signals based on market data.

While powerful, MLPs do not inherently understand the sequence of data, which can be a limitation for time-series analysis. For that, more specialized architectures like LSTMs are often preferred. However, MLPs are an excellent starting point and can be very effective when using carefully engineered features.
