import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set model parameters
d_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': 42,
}

# Train the model
model = lgb.train(d_params, train_data)

# Make predictions
predictions = model.predict(X_test)
# Convert probabilities to binary predictions
predictions_binary = np.round(predictions)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_binary)
print(f'Model accuracy: {accuracy:.4f}')