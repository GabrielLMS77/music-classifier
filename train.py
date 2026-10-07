import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)

print("Accuracy:", accuracy)

print("First 10 predictions:", predictions[:10])
print("First 10 actual:", y_test[:10])

accuracy = np.mean(predictions == y_test)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)

print("Confusion matrix:")
print(cm)