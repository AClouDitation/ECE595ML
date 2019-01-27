#! /usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import csv

def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    return fig, ax

def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')

if __name__ == "__main__":

    maleData = list() 
    femaleData = list() 
    # Reading csv file for male data
    with open("../data/male_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   # get the first row, header of data
        for row in reader:      # put data into list with the corresponding key
            maleData.append(row[1:])

    maleNum = len(maleData)
    csv_file.close()
    maleData = np.array(maleData, dtype=float)

    # Reading csv file for female data
    with open("../data/female_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   
        for row in reader:
            femaleData.append(row[1:])

    femaleNum = len(femaleData)
    csv_file.close()
    femaleData = np.array(femaleData, dtype=float)

    A = np.hstack((np.concatenate((maleData,femaleData), axis=0),np.ones((len(femaleData)+len(maleData),1))))
    A[:,1] /= 100   # to reduce the numerical error

    b = np.concatenate((np.ones((maleNum)), np.ones((femaleNum))*(-1)))

    # (8)
    
    N = len(A[0])
    lambda_list = np.arange(0.1, 10, 0.1)
    theta = list() 
    axmb = list()

    for lam in lambda_list:

        # construct problem
        x = cp.Variable(N)
        objective = cp.Minimize(cp.sum_squares(A*x-b) + lam*cp.sum_squares(x))
        constraints = [-20 <= x, x <= 20] # for now
        prob = cp.Problem(objective, constraints)

        # solve it
        result = prob.solve()
        theta.append(x.value)

        axmb.append(A.dot(x.value)-b)


    # Plot the decision boundaries with theta_{lambda=0.1, 1.1, ..., 9.1}
    fig, ax = newfig()
    x_tics = np.linspace(0, 100, 200)
    legend_str = []
    for i in range(len(lambda_list))[0::10]:
        y = -theta[i][2]/theta[i][1] - theta[i][0]/theta[i][1]*x_tics 
        ax.plot(x_tics, y.T)
        legend_str.append("$\lambda = $" + str(lambda_list[i]))
    plt.legend(legend_str)
    plt.xlabel("BMI")
    plt.ylabel("Stature")

    final_adjust("../pix/exercise4.pdf")
    
    # Plot ||Ax-b||^2 with respect to ||theta||^2
    fig, ax = newfig()
    ax.plot([np.sum(t**2) for t in theta], [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||Ax-b||^{2}_{2} vs ||\theta||^{2}_{2} $")
    ax.set_xlabel(r"$ ||\theta||^{2}_{2} $")
    ax.set_ylabel(r"$ ||Ax-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_1.pdf")

    # Plot ||Ax-b||^2 with respect to lambda
    fig, ax = newfig()
    ax.plot(lambda_list, [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||Ax-b||^{2}_{2} vs \lambda $")
    ax.set_xlabel(r"$ \lambda $")
    ax.set_ylabel(r"$ ||Ax-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_2.pdf")

    # Plot ||theta||^2 with respect to lambda
    fig, ax = newfig()
    ax.plot(lambda_list, [np.sum(t**2) for t in theta])
    ax.set_title(r"$ ||\theta||^{2}_{2} vs \lambda $")
    ax.set_xlabel(r"$ \lambda $")
    ax.set_ylabel(r"$ ||\theta_{\lambda}||^{2}_{2} $")
    final_adjust("../pix/exercise4_3.pdf")
