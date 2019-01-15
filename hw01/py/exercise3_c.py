import numpy as np
import matplotlib.pyplot as plt

def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    return fig, ax

def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


if __name__ == '__main__':

    mean = (0,0)
    cov = [[1,0],[0,1]]
    X = np.random.multivariate_normal(mean, cov, 5000)
    print(X.shape)
    fig, ax = newfig()
    ax.scatter(X[:,0],X[:,1])
    ax.grid(True)
    final_adjust('../pix/exercise3_d1.pdf')


    X = X.dot(np.array([[2**(0.5), 0],[2**(0.5)/2, 6**(0.5)/2]]))
    X[:,0] += 2
    X[:,1] += 6
    print(X.shape)
    fig, ax = newfig()
    ax.scatter(X[:,0],X[:,1])
    ax.grid(True)
    final_adjust('../pix/exercise3_d2.pdf')

    print(np.mean(X[:,0]))
    print(np.mean(X[:,1]))
    print(np.cov(X.T))
