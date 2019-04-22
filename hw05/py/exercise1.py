#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def __gee(x):
    return (x-2)**3+8*(x-2)**2+3


def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid()
    return fig, ax


def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


if __name__ == "__main__":

    x = np.arange(-7., 5., 0.2)
    fig, ax = newfig()
    ax.plot(x, __gee(x))
    ax.annotate('$x_0\'$',(0.1,__gee(0)+5))
    ax.plot(0, __gee(0), 'ro')
    ax.annotate('$x_0$',(4.1,__gee(4)+5))
    ax.plot(4, __gee(4), 'ro')
    ax.annotate('$x_{attack}$',(2.1,__gee(2)+5))
    ax.plot(2, __gee(2), 'bo')
    ax.axvline(x=-6, color='k', linestyle='--')
    ax.vlines(x=4, ymin = 0.0, ymax = __gee(4), color='k', linestyle='--')
    ax.axhline(y=0, color='k')
    ax.grid(True)
    final_adjust('../pix/exercise1_a2.pdf')
