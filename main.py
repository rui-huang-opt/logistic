import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from scipy.optimize import minimize

if __name__ == '__main__':
    n_train = 150
    n_test = 50
    n_samples = n_train + n_test
    n_features = 2
    n_nodes = 3

    # 生成二分类数据集
    X, y = make_classification(
        n_samples=n_samples,  # 样本数量
        n_features=n_features,  # 特征数量
        n_informative=n_features,  # 有效特征数量
        n_redundant=0,  # 冗余特征数量
        n_clusters_per_class=1,  # 每个类别的聚类数量
        random_state=51  # 随机种子
    )

    y = np.where(y == 0, -1, 1)

    X_train = X[:n_train, :]
    X_train_hat = np.hstack((X_train, np.ones((n_train, 1))))
    y_train = y[:n_train]

    X_test = X[n_train:, :]
    y_test = y[n_train:]

    C = 0.1

    def cost(theta):
        return np.sum(np.log(1 + np.exp(-y_train * (X_train_hat @ theta)))) / n_nodes + (C / 2) * np.sum(theta ** 2)

    results = minimize(cost, np.array([0., 0., 0.]))
    theta_star = results.x

    print(f'theta_star = {theta_star}')

    w = theta_star[:n_features]
    b = theta_star[-1]

    accuracy = np.sum(np.where(X_test @ w + b > 0, 1, -1) == y_test) / n_test
    print(f'accuracy = {accuracy}')

    # 可视化数据集
    fig1, ax1 = plt.subplots()
    ax1.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], color='red', label='Class 0', alpha=0.6)
    ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue', label='Class 1', alpha=0.6)

    ax1.set_title('Generated Binary Classification Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # 绘制决策边界
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # 计算边界上的y值
    x_boundary = np.linspace(x_min, x_max, 100)
    y_boundary = - (w[0] * x_boundary + b) / w[1]

    # 绘制决策边界
    ax1.plot(x_boundary, y_boundary, '--', color='green', label='Decision Boundary')

    ax1.legend()
    ax1.grid()

    plt.show()

    # split_X_train_hat = np.array_split(X_train_hat, n_nodes)
    # split_y_train = np.array_split(y_train, n_nodes)

    # for i, sub_X_train_hat in enumerate(split_X_train_hat):
    #     np.save(f'feature_node_{i + 1}.npy', sub_X_train_hat)
    #
    # for i, sub_y_train in enumerate(split_y_train):
    #     np.save(f'label_node_{i + 1}.npy', sub_y_train)
