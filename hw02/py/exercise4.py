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

    # (a)
    
    N = len(A[0])
    lambda_list = np.arange(0.1, 10, 0.1)
    theta = list() 
    axmb = list()

    for lam in lambda_list:

        # construct problem
        x = cp.Variable(N)
        objective = cp.Minimize(cp.sum_squares(A*x-b) + lam*cp.sum_squares(x))
        prob = cp.Problem(objective)

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
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs ||\theta||^{2}_{2} $")
    ax.set_xlabel(r"$ ||\theta||^{2}_{2} $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_1.pdf")

    # Plot ||Ax-b||^2 with respect to lambda
    fig, ax = newfig()
    ax.plot(lambda_list, [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs \lambda $")
    ax.set_xlabel(r"$ \lambda $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_2.pdf")

    # Plot ||theta||^2 with respect to lambda
    fig, ax = newfig()
    ax.plot(lambda_list, [np.sum(t**2) for t in theta])
    ax.set_title(r"$ ||\theta||^{2}_{2} vs \lambda $")
    ax.set_xlabel(r"$ \lambda $")
    ax.set_ylabel(r"$ ||\theta_{\lambda}||^{2}_{2} $")
    final_adjust("../pix/exercise4_3.pdf")


    # (c)
    theta_point_one = theta[0]

    # (i)
    
    alpha_star = sum([x**2 for x in theta_point_one])
    axmb = list()
    theta = list()

    alpha_list = [alpha_star + 2*i for i in range(-50,51)]
    for alpha in alpha_list:

        # construct problem
        x = cp.Variable(N)
        objective = cp.Minimize(cp.sum_squares(A*x-b) * 0.1)
        constraints = [cp.sum_squares(x) <= alpha] 
        prob = cp.Problem(objective, constraints)

        # solve it
        result = prob.solve()
        theta.append(x.value)
        axmb.append(A.dot(x.value)-b)

    # Plot ||Ax-b||^2 with respect to ||theta||^2
    fig, ax = newfig()
    ax.plot([np.sum(t**2) for t in theta], [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs ||\theta||^{2}_{2} $")
    ax.set_xlabel(r"$ ||\theta||^{2}_{2} $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_4.pdf")

    # Plot ||Ax-b||^2 with respect to alpha
    fig, ax = newfig()
    ax.plot(alpha_list, [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs \alpha $")
    ax.set_xlabel(r"$ \alpha $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_5.pdf")

    # Plot ||theta||^2 with respect to alpha
    fig, ax = newfig()
    ax.plot(alpha_list, [np.sum(t**2) for t in theta])
    ax.set_title(r"$ ||\theta||^{2}_{2} vs \alpha $")
    ax.set_xlabel(r"$ \alpha $")
    ax.set_ylabel(r"$ ||\theta_{\alpha}||^{2}_{2} $")
    final_adjust("../pix/exercise4_6.pdf")
        

    # (ii)
    
    epsilon_star = sum([x ** 2 for x in A.dot(theta_point_one)-b])

    axmb = list()
    theta = list()

    epsilon_list = [epsilon_star + 2*i for i in range(0,101)]

    for epsilon in epsilon_list:

        # construct problem
        x = cp.Variable(N)
        objective = cp.Minimize(cp.sum_squares(x))
        constraints = [cp.sum_squares(A*x-b) <= epsilon] 
        prob = cp.Problem(objective, constraints)

        # solve it
        result = prob.solve()
        theta.append(x.value)
        axmb.append(A.dot(x.value)-b)

    # Plot ||Ax-b||^2 with respect to ||theta||^2
    fig, ax = newfig()
    ax.plot([np.sum(t**2) for t in theta], [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs ||\theta||^{2}_{2} $")
    ax.set_xlabel(r"$ ||\theta||^{2}_{2} $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_7.pdf")

    # Plot ||Ax-b||^2 with respect to alpha
    fig, ax = newfig()
    ax.plot(epsilon_list, [np.sum(a**2) for a in axmb])
    ax.set_title(r"$ ||A\theta-b||^{2}_{2} vs \epsilon $")
    ax.set_xlabel(r"$ \epsilon $")
    ax.set_ylabel(r"$ ||A\theta-b||^{2}_{2} $")
    final_adjust("../pix/exercise4_8.pdf")

    # Plot ||theta||^2 with respect to epsilon
    fig, ax = newfig()
    ax.plot(epsilon_list, [np.sum(t**2) for t in theta])
    ax.set_title(r"$ ||\theta||^{2}_{2} vs \epsilon $")
    ax.set_xlabel(r"$ \epsilon $")
    ax.set_ylabel(r"$ ||\theta_{\epsilon}||^{2}_{2} $")
    final_adjust("../pix/exercise4_9.pdf")
