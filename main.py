import os
import numpy as np
import matplotlib.pyplot as plt
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

    fig1, ax1 = plt.subplots()
    ax1.scatter(
        X_train[y_train == -1, 0],
        X_train[y_train == -1, 1],
        color="red",
        label="Class -1",
        alpha=0.6,
    )
    ax1.scatter(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        color="blue",
        label="Class 1",
        alpha=0.6,
    )

    ax1.set_title("Generated Binary Classification Data")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    x_boundary = np.linspace(x_min, x_max, 100)
    y_boundary = -(w[0] * x_boundary + b) / w[1]

    ax1.plot(x_boundary, y_boundary, "--", color="green", label="Decision Boundary")

    ax1.legend()
    ax1.grid()

    figures_dir = os.path.join("figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig1.savefig(
        os.path.join(figures_dir, "decision_boundary.png"), dpi=300, bbox_inches="tight"
    )

    data_dir = "data"
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
