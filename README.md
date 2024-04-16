# Heart-Disease-Prediction
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load your dataset
# df = pd.read_csv('heart_disease_data.csv')

# Let's assume 'df' is your DataFrame and 'target' is the column with the disease status
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% testing

# Create a Logistic Regression model object
log_regression = LogisticRegression()

# Train the model using the training sets
log_regression.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = log_regression.predict(X_test)

# Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
