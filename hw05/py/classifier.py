#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import log
import scipy.ndimage
import sys

def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid()
    return fig, ax


def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')


def task_a():
    train_cat = np.matrix(np.loadtxt('../data/train_cat.txt', delimiter=','))
    train_grass = np.matrix(np.loadtxt('../data/train_grass.txt', delimiter=','))

    mu_cat      = np.asmatrix(np.mean(train_cat, 1))
    mu_grass    = np.asmatrix(np.mean(train_grass, 1))

    cov_cat     = np.asmatrix(np.cov(train_cat))
    cov_grass   = np.asmatrix(np.cov(train_grass))

    pi_cat      = len(train_cat.T) / (len(train_cat.T) + len(train_grass.T))
    pi_grass    = len(train_grass.T) / (len(train_cat.T) + len(train_grass.T))

    return (mu_cat, cov_cat, pi_cat), (mu_grass, cov_grass, pi_grass)


def __gee(x, info):
    return -1/2 * np.matmul(np.matmul((x-info[0]).T,np.linalg.inv(info[1])),(x-info[0]))


def classify(pix, outputFn, overlapping=True):

    fig,ax = newfig()
    Y = scipy.ndimage.imread(pix, mode='L') /255
    M,N = Y.shape
    cat_info, grass_info = task_a()
    const_cat   = -log(np.linalg.det(cat_info[1])**(1/2))
    const_grass = -log(np.linalg.det(grass_info[1])**(1/2))
    
    Output = np.zeros((M,N))
    range_i = range(M-8)
    range_j = range(N-8)
    if not overlapping:
        range_i = range(0,M-8,8)
        range_j = range(0,N-8,8)

    for i in range_i:
        for j in range_j:
            z = Y[i:i+8, j:j+8]
            z_vector = np.asmatrix(z.flatten('F')).T
            pcat    = const_cat     +__gee(z_vector, cat_info)  +log(cat_info[2]) 
            pgrass  = const_grass   +__gee(z_vector, grass_info)+log(grass_info[2]) 
            if pcat > pgrass:
                Output[i][j] = 1
                if not overlapping:
                    for ii in range(8):
                        for jj in range(8):
                            Output[i+ii][j+jj] = 1

    plt.imshow(Output*255, cmap='gray')
    final_adjust(outputFn)
    return Output


def MAE(output, ground_truth):
    return np.sum(np.absolute(output-ground_truth)) / ground_truth.size


if __name__ == "__main__":

    #overlapping_output = classify('../data/cat_grass.jpg','../pix/overlapping_output.pdf')
    #non_overlapping_output = classify('../data/cat_grass.jpg','../pix/non_overlapping_output.pdf', overlapping=False)
    test_output = classify(sys.argv[1],'../pix/cwattack.pdf', overlapping=False)

    ground_truth = plt.imread('../data/truth.png')
    #print("overlapping MAE: %.4f"%(MAE(overlapping_output, ground_truth)))
    #print("non-overlapping MAE: %.4f"%(MAE(non_overlapping_output, ground_truth)))
    print("after attack MAE: %.4f"%(MAE(test_output, ground_truth)))

    


