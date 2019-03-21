#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid()
    return fig, ax


def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


def drawline(ax, theta):
    slope = -theta[0] / theta[1]
    intercept = -theta[2] / theta[1]

    x1 = [-0.5, 0.5]
    x2 = [-0.5 * slope+intercept, 0.5 * slope+intercept]

    ax.plot(x1, x2, '-')


def import_data():

    samples = np.array(np.loadtxt('../data/hw04_sample_vectors.csv', delimiter=','))
    labels = np.array(np.loadtxt('../data/hw04_labels.csv', delimiter=','))

    # fig, ax = newfig()
    # ax.plot(samples[:,0], samples[:,1], '.')
    # plt.show()

    return samples, labels


def logistic(X, labels, learning_rate=0.1, max_num_iterations=200):
    
    def hypfunc(theta, X):
        return 1 / (1 + np.exp(-np.dot(theta, np.concatenate([X, [1]]).T)))
    

    def cost_function_derivative(X, labels, theta):
        return sum([(hypfunc(theta, X[i]) - labels[i])*np.concatenate([X[i], [1]]) for i in range(len(X))])


    theta = np.array([0,0,0], dtype=np.float64)

    fig, ax = newfig()
    lenl1 = int(sum(labels))
    lenl0 = len(labels) - lenl1

    ax.plot(X[:lenl0,0], X[:lenl0,1], '.')
    ax.plot(X[lenl1:,0], X[lenl1:,1], '.')
    ax.axis([-1.3,1.3,-0.7,0.5])
    for m in range(max_num_iterations):
        theta -= learning_rate*cost_function_derivative(X, labels, theta)
        if m % (max_num_iterations / 5) == 0: 
            drawline(ax, theta)

    drawline(ax, theta)
    final_adjust('../pix/logistic.pdf')


if __name__ == "__main__":

    X, labels = import_data()
    logistic(X, labels)
