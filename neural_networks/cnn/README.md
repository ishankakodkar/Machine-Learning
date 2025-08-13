# Convolutional Neural Network (CNN) for Financial Time Series

Convolutional Neural Networks (CNNs or ConvNets), while traditionally famous for their performance in computer vision, are also highly effective for sequence data, including financial time series. By using 1D convolutional layers, CNNs can learn to recognize patterns (like bullish or bearish chart patterns) over a fixed window of time-series data.

## Mathematical Formulation

A 1D CNN applies a set of learnable filters (or kernels) to a sequence of data. Each filter slides across the sequence, performing a convolution operation to create a feature map that highlights specific patterns.

### 1. The Convolution Operation (1D)

A filter is a small vector of weights. The convolution operation involves sliding this filter over the input sequence and, at each position, computing the dot product between the filter and the portion of the sequence it is currently covering.

For an input sequence \( S \) and a filter \( F \) of size \( k \), the output feature map \( O \) at position \( t \) is:

\[ O_t = g \left( \sum_{i=1}^{k} S_{t+i-1} \cdot F_i + b \right) \]

Where:
-   \( g \) is an activation function (typically ReLU).
-   \( b \) is a bias term.

The filter's weights are learned during training via backpropagation. The network learns filters that activate when they detect specific patterns (e.g., a sharp increase, a period of low volatility) in the input data.

### 2. Pooling Layers

After a convolution, it is common to use a pooling layer (e.g., **Max Pooling**) to downsample the feature map. A 1D max pooling layer slides a window over the feature map and, for each window, outputs only the maximum value.

Pooling serves two main purposes:
1.  **Reduces Dimensionality:** It makes the representation smaller and more manageable.
2.  **Creates Translational Invariance:** It makes the network robust to the exact position of a pattern in the input sequence. A pattern detected at the beginning of a window will produce the same output as one detected at the end.

### 3. Architecture

A typical 1D CNN for time series consists of:
1.  One or more `Conv1D` and `MaxPooling1D` layers to form the convolutional base for feature extraction.
2.  A `Flatten` layer to convert the 2D feature maps into a 1D vector.
3.  One or more `Dense` layers (an MLP) to interpret the extracted features and make a final prediction.

## Application in Finance

-   **Pattern Recognition:** Automatically detect and learn technical chart patterns (e.g., head and shoulders, flags) from raw price data.
-   **Volatility Forecasting:** Predict future volatility by learning from patterns in historical price changes.
-   **Signal Generation:** Classify a sequence of market data as a "buy," "sell," or "hold" signal.
