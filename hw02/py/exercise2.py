#! usr/bin/python3

import numpy as np
import cvxpy as cp
import csv

if __name__ == "__main__":

    
    data = list() 
    # Reading csv file for male data
    with open("../data/male_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   # get the first row, header of data

        for row in reader:      # put data into list with the corresponding key
            data.append(row[1:])

    maleNum = len(data)
    csv_file.close()

    # Reading csv file for female data
    with open("../data/female_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   
        for row in reader:
            data.append(row[1:])

    csv_file.close()
    femaleNum = len(data) - maleNum

    A = np.hstack((np.array(data, dtype=float),np.ones((len(data),1))))
    b = np.concatenate((np.ones((maleNum)), np.ones((femaleNum))*(-1)))

    # (c) using calculus
    theta = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    print(theta)

    # (d) using cvxpy
    
    # construct problem
    N = len(A[0])
    x = cp.Variable(N)
    objective = cp.Minimize(cp.sum_squares(A*x-b))
    constraints = [-20 <= x, x <= 20] # for now
    prob = cp.Problem(objective, constraints)

    # solve it
    result = prob.solve()
    print(x.value)



