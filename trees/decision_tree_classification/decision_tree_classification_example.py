import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# 1. Load the dataset
dataset = pd.read_csv('trading_signals.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train the Decision Tree model
# We use a shallow tree (max_depth=3) to keep the rules simple and interpretable
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Trading Model")
print("===========================")
print(f"Model Accuracy: {accuracy*100:.2f}%")

# 5. Display the trading rules
print("\n--- Interpretable Trading Rules ---")
feature_names = list(X.columns)
tree_rules = export_text(classifier, feature_names=feature_names)
print(tree_rules)

# 6. Make a prediction on a new data point
# Example: [RSI, MACD_Signal, Bollinger_Band_Signal]
new_data = [[65, 1, 0]] # RSI is high, MACD is positive, Bollinger is neutral
prediction = classifier.predict(new_data)

print("\n--- New Data Point Prediction ---")
print(f"Input Signals: RSI=65, MACD=Positive, Bollinger=Neutral")
if prediction[0] == 1:
    print("Predicted Action: BUY")
else:
    print("Predicted Action: SELL/HOLD")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Load the dataset
dataset = pd.read_csv('social_network_ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree Classification model on the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Visualising the Training set results
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
