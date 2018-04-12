import numpy as np

import matplotlib.pyplot as plt


def draw(X_, y_):
    plt.scatter(X_, y_)
    plt.axvline(x=0)
    plt.grid(True)
    plt.show()


x = np.arange(1, 11)
y = 2 * x + np.random.randn(10) * 2
X = np.vstack((x, y))
draw(x, y)

Xcentered = (X[0] - x.mean(), X[1] - y.mean())
m = (x.mean(), y.mean())
draw(*Xcentered)

covmat = np.cov(Xcentered)
print(covmat, "\n")
print("Variance of X: ", np.cov(Xcentered)[0, 0])
print("Variance of Y: ", np.cov(Xcentered)[1, 1])
print("Covariance X and Y: ", np.cov(Xcentered)[0, 1])
print("Covariance X and Y: ", np.cov(Xcentered)[1, 0])
