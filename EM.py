import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import multivariate_normal

def plot_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.3)
    ax.add_patch(ell)

def em_algorithm(data, num_components, max_iter=100, tol=1e-4):
    num_samples, num_features = data.shape
    responsibilities = np.zeros((num_samples, num_components))
    weights = np.ones(num_components) / num_components
    means = np.random.rand(num_components, num_features)
    covariances = np.array([np.eye(num_features) for _ in range(num_components)])

    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step
        for i in range(num_components):
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covariances[i])

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step
        Nk = responsibilities.sum(axis=0)
        weights = Nk / num_samples
        means = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]

        for i in range(num_components):
            diff = data - means[i]
            covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]

        # Calculate log likelihood
        log_likelihood = np.sum(np.log(np.sum(weights[i] * multivariate_normal.pdf(data, means[i], covariances[i])
                                              for i in range(num_components))))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iteration > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tol:
            break

    return means, covariances, weights, responsibilities, log_likelihoods

iris = datasets.load_iris()
data = iris.data

num_components = 3
means, covariances, weights, responsibilities, log_likelihoods = em_algorithm(data, num_components)

fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], c=iris.target, cmap='viridis', marker='o', alpha=0.8)

for i in range(num_components):
    plot_ellipse(ax, means[i, :2], covariances[i, :2, :2], color='red')

plt.title('Expectation-Maximization (EM) Algorithm on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
