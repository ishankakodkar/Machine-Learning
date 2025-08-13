# Encoder-Decoder (Seq2Seq) for Multi-Step Forecasting

The Encoder-Decoder, or Sequence-to-Sequence (Seq2Seq), architecture is a powerful type of neural network designed for tasks where the input and output are both sequences of variable length. While it originated in natural language processing (e.g., for machine translation), it is highly effective for complex multi-step time-series forecasting in finance.

## Conceptual Framework

The architecture consists of two main components: an **Encoder** and a **Decoder**, which are typically implemented using recurrent layers like LSTMs or GRUs.

### 1. The Encoder

The encoder's job is to process the entire input sequence and compress it into a fixed-size representation called the **context vector** (or "thought vector"). This vector aims to capture the essential information of the input sequence.

-   An LSTM (or GRU) layer reads the input sequence one step at a time.
-   The final hidden state (\( h_t \)) and cell state (\( C_t \)) of the LSTM after it has seen the entire input sequence are used as the context vector. These states serve as a numerical summary of the input sequence.
-   The outputs of the encoder at each time step are typically discarded; only the final states are kept.

\[ \text{context} = [h_{\text{final}}, C_{\text{final}}] = \text{Encoder}(\text{input sequence}) \]

### 2. The Decoder

The decoder's job is to take the context vector from the encoder and generate the output sequence one step at a time.

-   The context vector is used as the initial hidden state and cell state of the decoder's LSTM layer.
-   At each step, the decoder takes the state from the previous step and generates an output. This output is then fed back as the input for the next step, allowing it to generate a sequence.
-   In a forecasting context, the decoder is trained to predict a sequence of future values (e.g., the next 5 days of stock prices).

### Implementation in Keras

-   **`RepeatVector`**: This layer is used to feed the final context vector from the encoder to the decoder at each time step of the output sequence. It simply repeats the context vector `n` times, where `n` is the length of the output sequence.
-   **`TimeDistributed`**: This wrapper applies a layer (e.g., a `Dense` layer) to every temporal slice of an input. It is used on the output of the decoder's LSTM to generate a prediction at each time step.

## Application in Finance

-   **Multi-Step Ahead Price Forecasting:** Predict a sequence of future stock prices (e.g., prices for the next 5 days) based on a sequence of historical prices.
-   **Volatility Term Structure:** Forecast the entire forward volatility curve based on past market conditions.
-   **Portfolio Trajectory:** Predict the future value path of a portfolio under different scenarios.
