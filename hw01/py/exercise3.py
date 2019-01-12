import numpy as np
import matplotlib.pyplot as plt

def newfig():
    fig = plt.figure(figsize=(6,9), dpi=300)
    ax = fig.add_subplot(111)
    return fig, ax

def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


if __name__ == '__main__':
    x1 = np.arange(-1,5,0.01)
    x2 = np.arange(0,10,0.01)
    Z = np.zeros((1000,600))
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j][i] = 1/(np.pi*6**(1/2))*np.exp(-1/3*((x1[i]-2)**2
                     -(x1[i]-2)*(x2[j]-6)+(x2[j]-6)**2))

    fig, ax = newfig()
    ax.contour(x1,x2,Z)
    ax.grid(True)
    ax.set_yticks([i for i in range(11)])
    ax.set_xticks([i for i in range(-1,6)])
    ax.set_xlabel('x1', fontsize = 20)
    ax.set_ylabel('x2', fontsize = 20)

    final_adjust('../pix/exercise3_a.pdf')

