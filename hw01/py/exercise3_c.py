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
    X = np.random.multivariate_normal(mean, cov, 5000).T
    fig, ax = newfig()
    ax.scatter(X[0,:],X[1,:])
    ax.grid(True)
    final_adjust('../pix/exercise3_c1.pdf')


    A = np.array([[2**(0.5), 0],[2**(0.5)/2, 6**(0.5)/2]])
    
    print(X.shape)
    X = A.dot(X)
    X[0,:] += 2
    X[1,:] += 6
    fig, ax = newfig()
    ax.scatter(X[0,:],X[1,:])
    ax.grid(True)
    final_adjust('../pix/exercise3_c2.pdf')

    print(np.mean(X[0,:]))
    print(np.mean(X[1,:]))
    print(np.cov(X))

    Sigma = np.array([[2,1],[1,2]])
    
    w, v = np.linalg.eig(Sigma)
    X = np.random.multivariate_normal(mean, cov, 5000).T
    X = (v*w**(1/2)).dot(X)

    X[0,:] += 2
    X[1,:] += 6
    fig, ax = newfig()
    ax.scatter(X[0,:],X[1,:])
    ax.grid(True)
    final_adjust('../pix/exercise3_c3.pdf')
