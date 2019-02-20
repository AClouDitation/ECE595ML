#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp


def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid()
    return fig, ax


def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


if __name__ == "__main__":

    # generate data
    mu_1 = np.array([0,0])
    mu_2 = np.array([20,20])
    cov = np.array([[10,0], [0,20]])

    X_1 = np.random.multivariate_normal(mu_1, cov, 1000)
    X_2 = np.random.multivariate_normal(mu_2, cov, 1000)

    fig, ax = newfig()
    ax.plot(X_1[:,0], X_1[:,1], 'x')
    ax.plot(X_2[:,0], X_2[:,1], '.')

    # plot decision boundrary
    sigma_inv = np.linalg.inv(cov) 
    beta = sigma_inv.dot(mu_1 - mu_2)
    beta_0 = -1/2*(np.matmul(np.matmul(mu_1.T,sigma_inv), mu_1) - 
            np.matmul(np.matmul(mu_2.T,sigma_inv), mu_2))

    slope = -beta[0]/beta[1]
    intercept = -beta_0/beta[1]

    x = np.arange(-10, 30)
    y = x * slope + intercept
    ax.plot(x, y)

    # solve least square
    A = np.concatenate((X_1,X_2))
    A = np.hstack((A,np.ones((len(A),1))))
    b = np.concatenate((np.ones(1000), -np.ones(1000)))

    # construct problem
    N = len(A[0])
    x = cp.Variable(N)
    objective = cp.Minimize(cp.sum_squares(A*x-b))
    prob = cp.Problem(objective)

    # solve it
    result = prob.solve()

    theta = x.value
    slope = -theta[0]/theta[1]
    intercept = -theta[2]/theta[1]

    x = np.arange(-10, 30)
    y = x * slope + intercept
    ax.plot(x, y)
    plt.show()

