#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def my_training(train_cat, train_grass):
    W = [None] * 2
    w = [None] * 2
    w0 = [None] * 2
    # calculate the mean of cat class and grass class
    mu_grass = np.mean(train_grass, 1)
    mu_cat = np.mean(train_cat, 1)
    # calculate the sample priors
    pi_cat = len(train_cat[0]) / (len(train_cat[0]) + len(train_grass[0]))
    pi_grass = len(train_grass[0]) / (len(train_cat[0]) + len(train_grass[0]))

    # calculate the W matrix of cat and grass classes
    W[0] = np.linalg.inv(np.cov(train_grass))
    W[1] = np.linalg.inv(np.cov(train_cat))

    # calculate the w vector for cat and grass classes
    w[0] = -np.matmul(W[0], mu_grass)
    w[1] = -np.matmul(W[1], mu_cat)

    w0[0] = np.matmul(np.matmul(mu_grass, W[0]), mu_grass) / 2 + np.log(np.linalg.det(np.cov(train_grass))) / 2 - np.log(pi_grass)
    w0[1] = np.matmul(np.matmul(mu_cat, W[1]), mu_cat) / 2 + np.log(np.linalg.det(np.cov(train_cat))) / 2 - np.log(pi_cat)

    # return calculated numpy arrays (matrices)
    return W, w, w0


def gradient(x, x0, lamb, target_idx, W, w, w0):
    # x: 	the current vector			x0:	the original vector
    # lamb: lambda value				W:  the Wj and Wt matrix
    # w:    the wj and wt vector        w0: the w_0j and w_0t scalar
    # target idx = 0 --> make grass into cat
    # target idx = 1 --> make cat into grass
    j = 1 - target_idx
    t = target_idx
    # caluclate g(x) see if it is already in target class
    g = np.matmul(np.matmul(x.T, W[j] - W[t]), x) / 2 + np.matmul(x.T, w[j] - w[t]) + w0[j] - w0[t]
    if g > 0:
        return [0] * 64
    # if it is not in target class, calculate the gradient and return it
    grad = -2 * (x - x0) - lamb * np.matmul(W[j] - W[t], x) + w[j] - w[t]
    return grad


def nonoverlapping_CW_attack(Y, lamb, alpha, target_idx, W, w, w0):
    # initialize the index and change to start the loop
    idx = 0; change = 99

    # M and N for the dimension of the input image
    M = len(Y); N = len(Y[0])

    # an err vector to track how many mis classified pixels
    err = []

    # use prev and next to track the change
    prev = np.copy(Y)
    next = np.zeros((M, N))

    while idx <= 300 and change >= 0.001:
        for i in range(M // 8):
            for j in range(N // 8):
                # get the patch from both original image and attacked image
                x = prev[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8].flatten('F')
                x0 = Y[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8].flatten('F')
                # calculate the gradient
                grad = gradient(x, x0, lamb, target_idx, W, w, w0)
                # loop through all pixels and change the value based on the gradient
                for ii in range(8):
                    for jj in range(8):
                        next[i * 8 + ii][j * 8 + jj] = np.clip(x[jj * 8 + ii] - alpha * grad[jj * 8 + ii], 0.0, 1.0)
        # increment the index and calculate the differences between two steps
        idx += 1
        change = np.linalg.norm((next - prev).flatten('F'))
        prev = np.copy(next)
        # calculate error for every 5 iteration
        if idx % 5 == 0:
            err.append(np.count_nonzero(my_testing_nonoverlap(next, W, w, w0)))
            print(f'iteration {idx}: {err[-1]}')
    # if attacked finished early before 100 iterations, append error vector with zero
    for _ in range(20 - len(err)):
        err.append(0)
    print(f'Finished in {idx} iterations')
    return next, err


def overlapping_CW_attack(Y, lamb, alpha, target_idx, W, w, w0):
    # initialize the index and change to start the loop
    idx = 0; change = 99

    # M and N for the dimension of the input image
    M = len(Y); N = len(Y[0])

    prev = np.copy(Y)
    next = np.zeros((M, N))

    while idx <= 300 and change >= 0.01:

        for i in range(M - 8):
            for j in range(N - 8):
                x0 = Y[i : i + 8, j : j + 8].flatten('F')
                x = prev[i : i + 8, j : j + 8].flatten('F')
                grad = gradient(x, x0, lamb, target_idx, W, w, w0)
                for ii in range(8):
                    for jj in range(8):
                        next[i + ii][j + jj] = np.clip(x[jj * 8 + ii] - alpha * grad[jj * 8 + ii], 0.0, 1.0)

        idx += 1
        change = np.linalg.norm((next - prev).flatten('F'))
        prev = np.copy(next)
        print(f'{idx}: {change}')
    return next


def my_testing_overlap(Y, W, w, w0):
    # find the dimension of the image
    M = len(Y)
    N = len(Y[0])

    output = np.zeros((M - 8, N - 8))   # initialize the output matrix
    for i in range(M - 8):
        for j in range(N - 8):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = Y[i : i + 8, j : j + 8]
            z = z.flatten('F')
            # calculate g(x) and classify the pixel based on sign(g(x))
            g = np.matmul(np.matmul(z.T, W[0] - W[1]), z) / 2 + np.matmul(z.T, w[0] - w[1]) + w0[0] - w0[1]
            if g > 0:
                output[i][j] = 1
    return output


def my_testing_nonoverlap(Y, W, w, w0):
    # find the dimension of the image
    M = len(Y)
    N = len(Y[0])

    output = np.zeros((M, N))   # initialize the output matrix
    for i in range(M // 8):
        for j in range(N // 8):
            # extract the 8x8 patch, flatten it and find the difference to the mu of cat and grass
            z = Y[i *  8: i * 8 + 8, j * 8 : j * 8 + 8]
            z = z.flatten('F')
            # calculate g(x) and classify the batch based on sign(g(x))
            g = np.matmul(np.matmul(z.T, W[0] - W[1]), z) / 2 + np.matmul(z.T, w[0] - w[1]) + w0[0] - w0[1]
            if g > 0:
                for ii in range(8):
                    for jj in range(8):
                        output[i * 8 + ii][j * 8 + jj] = 1
    return output


def analize_CW(Y, CW, lamb, W, w, w0):
    M = len(Y) // 8 * 8
    N = len(Y[0]) // 8 * 8
    plt.imsave(f'CW_Nonoverlap_{lamb}.jpg', CW * 255, cmap='gray')
    diff = np.fabs(Y - CW)

    plt.imsave(f'CW_Patch_{lamb}.jpg', diff * 255, cmap='gray')
    frobenius = np.linalg.norm(diff)
    print('Frobenius:', frobenius)
    classify = my_testing_nonoverlap(CW, W, w, w0)
    print('Result:', np.count_nonzero(classify))
    print('-------------------------------')


def exercise2(Y, W, w, w0):
    CUT = []
    M = len(Y) // 8 * 8
    N = len(Y[0]) // 8 * 8

    for row in Y[:M]:
        CUT.append(row[:N])
    CUT = np.array(CUT)

    CW_1, err_1 = nonoverlapping_CW_attack(CUT, 1, 0.0001, 0, W, w, w0)
    analize_CW(CUT, CW_1, 1, W, w, w0)

    CW_5, err_5 = nonoverlapping_CW_attack(CUT, 5, 0.0001, 0, W, w, w0)
    analize_CW(CUT, CW_5, 5, W, w, w0)

    CW_10, err_10 = nonoverlapping_CW_attack(CUT, 10, 0.0001, 0, W, w, w0)
    analize_CW(CUT, CW_10, 10, W, w, w0)

    x = np.linspace(5, 100, 20)
    plt.plot(x, err_1, color='blue', label='$\lambda=1$')
    plt.plot(x, err_5, color='red', label='$\lambda=5$')
    plt.plot(x, err_10, color='black', label='$\lambda=10$')
    plt.xlabel('number of iterations')
    plt.ylabel('number of pixels classified as cat')
    plt.legend()
    plt.title('Number of cat pixels vs. number of iterations across different $\lambda$')
    plt.savefig('error_vs_iter.eps')


def exercise3(Y, W, w, w0):
    output = overlapping_CW_attack(Y, 10, 0.0002, 0, W, w, w0)
    plt.imsave('overlapping_CW.jpg', output * 255, cmap='gray')
    output = my_testing_overlap(output, W, w, w0)
    plt.imsave('overlapping_result.jpg', output * 255, cmap='gray')
    pass


if __name__ == '__main__':
    # read the input image and training data, then calculate all paramters
    Y = plt.imread('../data/cat_grass.jpg') / 255
    train_cat = np.loadtxt('../data/train_cat.txt', delimiter=',')
    train_grass = np.loadtxt('../data/train_grass.txt', delimiter=',')
    W, w, w0 = my_training(train_cat, train_grass)

    print(W[0])
    print(w[0])
    print(w0[0])

    exercise2(Y, W, w, w0)
    #exercise3(Y, W, w, w0)

