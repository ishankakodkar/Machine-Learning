import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Load the dataset
dataset = pd.read_csv('market_data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Training the SVM model
# We use the RBF kernel, which is effective for non-linear relationships
classifier = SVC(kernel='rbf', random_state=0, probability=True)
classifier.fit(X_train, y_train)

# 5. Predicting the Test set results
y_pred = classifier.predict(X_test)

# 6. Evaluating the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("SVM for Market Direction Prediction")
print("=================================")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# 7. Predicting a new observation
# Example: [Feature1, Feature2, Feature3] (e.g., lagged returns, volatility)
new_data = sc.transform([[0.01, -0.02, 0.03]])
prediction = classifier.predict(new_data)
prediction_proba = classifier.predict_proba(new_data)

print("\n--- New Data Point Prediction ---")
print(f"Prediction: Market will go {prediction[0]}")
print(f"Probability of 'Down': {prediction_proba[0][0]:.4f}")
print(f"Probability of 'Up': {prediction_proba[0][1]:.4f}")
