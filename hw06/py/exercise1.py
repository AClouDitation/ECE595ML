#! /usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt

def newfig():
    axs = []
    fig = plt.figure(figsize=(27,6), dpi=300)
    axs.append(fig.add_subplot(131))
    axs.append(fig.add_subplot(132))
    axs.append(fig.add_subplot(133))
    for ax in axs:
        ax.grid()
    return fig, axs


def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


if __name__ == "__main__":

    SIZE = 100000
    v_1s    = np.zeros([SIZE])
    v_rands = np.zeros([SIZE])
    v_mins  = np.zeros([SIZE])

    for r in range(SIZE):
        if(r % 1000 == 0):
            print(r)
        sums = np.random.binomial(10,0.5,1000)
        v_1     = sums[0] / 10
        v_rand  = sums[random.randint(0,999)] / 10
        v_min   = np.min(sums) / 10

        v_1s[r]     = v_1
        v_rands[r]  = v_rand
        v_mins[r]   = v_min

    print(np.mean(v_mins) * 10)
    print("DONE!")
    fig, axs = newfig()
    axs[0].hist(v_1s)
    axs[0].set_title('$v_1$')
    axs[1].hist(v_rands)
    axs[1].set_title('$v_{rand}$')
    axs[2].hist(v_mins)
    axs[2].set_title('$v_{min}$')
    final_adjust("../pix/exercise1_b.pdf")


    epsilon = np.arange(0, 0.5, 0.05)
    hoeffding = 2 * np.exp(-2 * epsilon ** 2 * 10)
    # P(|v1-mu1|>e)
    pv1     = [np.sum(np.abs(v_1s - 0.5) > epsilon[i]) / len(v_1s) for i in range(len(epsilon))]
    # P(|vrand-murand|>e)
    pvrand  = [np.sum(np.abs(v_rands - 0.5) > epsilon[i]) / len(v_1s) for i in range(len(epsilon))]
    # P(|vmin-mumin|>e)
    pvmin   = [np.sum(np.abs(v_mins - 0.5) > epsilon[i]) / len(v_1s) for i in range(len(epsilon))]


    fig, axs = newfig()
    axs[0].plot(epsilon, pv1)
    axs[0].plot(epsilon, hoeffding)
    axs[0].set_title('$\mathbb{P}(v_1-\mu_1 > \epsilon)$')
    axs[1].plot(epsilon, pvrand)
    axs[1].plot(epsilon, hoeffding)
    axs[1].set_title('$\mathbb{P}(v_{rand}-\mu_{rand} > \epsilon)$')
    axs[2].plot(epsilon, pvmin)
    axs[2].plot(epsilon, hoeffding)
    axs[2].set_title('$\mathbb{P}(v_{min}-\mu_{min} > \epsilon)$')
    final_adjust("../pix/exercise1_c.pdf")
    plt.show()
