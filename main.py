import os
import numpy as np
from sklearn.datasets import make_classification
from scipy.optimize import minimize

if __name__ == "__main__":
    n_train = 800
    n_test = 50
    n_samples = n_train + n_test
    n_features = 2
    n_nodes = 8

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=54,
    )

    y = np.where(y == 0, -1, 1)

    X_train = X[:n_train, :]
    X_train_hat = np.hstack((X_train, np.ones((n_train, 1))))
    y_train = y[:n_train]

    X_test = X[n_train:, :]
    y_test = y[n_train:]

    C = 0.1

    def cost(theta):
        return np.sum(
            np.log(1 + np.exp(-y_train * (X_train_hat @ theta)))
        ) / n_nodes + (C / 2) * np.sum(theta**2)

    results = minimize(cost, np.array([0.0, 0.0, 0.0]))
    theta_star = results.x

    print(f"theta_star = {theta_star}")

    w = theta_star[:n_features]
    b = theta_star[-1]

    accuracy = np.sum(np.where(X_test @ w + b > 0, 1, -1) == y_test) / n_test
    print(f"accuracy = {accuracy}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Create train directories and save data
    for i, (sub_X_train_hat, sub_y_train) in enumerate(
        zip(np.array_split(X_train_hat, n_nodes), np.array_split(y_train, n_nodes))
    ):
        train_dir = os.path.join(data_dir, f"train_{i + 1}")
        os.makedirs(train_dir, exist_ok=True)
        np.save(os.path.join(train_dir, "feature.npy"), sub_X_train_hat)
        np.save(os.path.join(train_dir, "label.npy"), sub_y_train)

    # Create test directory and save data
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    np.save(os.path.join(test_dir, "feature.npy"), X_test)
    np.save(os.path.join(test_dir, "label.npy"), y_test)
