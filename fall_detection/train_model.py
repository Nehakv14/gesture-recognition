import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
data = pd.read_csv('hand_landmarks.csv', header=None)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data for training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Test model
accuracy = model.score(X_test, y_test)
print(f"✅ Training complete. Accuracy: {accuracy * 100:.2f}%")

# Save trained model to file
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as gesture_model.pkl")
