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
    b = np.concatenate((np.ones((maleNum)), np.ones((femaleNum))*(-1)))

    theta = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)

    print(theta)

    
    #(a)
    fig, ax = newfig()
    ax.scatter(maleData[:,0], maleData[:,1])
    ax.scatter(femaleData[:,0], femaleData[:,1])
    ax.plot([0, 100], [-theta[2]/theta[1],-theta[0]*100/theta[1]-theta[2]/theta[1]])

    final_adjust("../pix/exercise3.pdf")

    maletestData = list() 
    femaletestData = list() 

    #(b)
    # Reading csv file for male data
    with open("../data/male_test_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   # get the first row, header of data
        for row in reader:      # put data into list with the corresponding key
            maletestData.append(row[1:])

    csv_file.close()
    maletestData = np.array(maletestData, dtype=float)

    # Reading csv file for female data
    with open("../data/female_test_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        header = next(reader)   
        for row in reader:
            femaletestData.append(row[1:])

    csv_file.close()
    femaletestData = np.array(femaletestData, dtype=float)
    
    succMale = sum([(theta[0]*x[0]+theta[1]*x[1]+theta[2]) >= 0 for x in maletestData])
    succFemale = sum([(theta[0]*x[0]+theta[1]*x[1]+theta[2]) < 0 for x in femaletestData])

    rate = (succMale+succFemale)/(len(maletestData)+len(femaletestData))
    print(rate)
