import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load the dataset
dataset = pd.read_csv('technical_indicators.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Build the MLP model
model = Sequential([
    Dense(units=6, activation='relu', input_dim=X_train.shape[1]), # Input layer and first hidden layer
    Dense(units=6, activation='relu'),                          # Second hidden layer
    Dense(units=1, activation='sigmoid')                        # Output layer
])

# 5. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the model
print("Training the model...")
model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1)

# 7. Evaluate the model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# 8. Making a prediction
print("\nMaking a sample prediction...")
sample_data = np.array([[55, 60, 45, 2500000]]) # Example: MA10, MA50, RSI, Volume
sample_data_scaled = sc.transform(sample_data)
prediction = model.predict(sample_data_scaled)

print(f'Prediction for sample data: {prediction[0][0]:.4f}')
if prediction[0][0] > 0.5:
    print("Prediction: Stock price will go UP")
else:
    print("Prediction: Stock price will go DOWN")
