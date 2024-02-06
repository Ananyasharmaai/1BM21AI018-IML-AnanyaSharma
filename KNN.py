#KNN algorithm
import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[2, 3]])

knn = KNN(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("Predictions:", predictions)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c='r', marker='x', label='Test Data')

for i in range(len(X_test)):
    plt.plot([X_test[i, 0]] * knn.k, [X_test[i, 1]] * knn.k, 'k--', alpha=0.5)

plt.title('KNN Algorithm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
