#! usr/bin/python3

import cvxpy as cp
import numpy as np
import csv


if __name__ == "__main__":

    # Reading csv file for male data
    with open("../data/male_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        maleData = dict()
        header = next(reader)   # get the first row, header of data
        for name in header:     # create a dictionary of lists with names in header as keys
            maleData[name] = list()

        for row in reader:      # put data into list with the corresponding key
            for i in range(len(row)):
                maleData[header[i]].append(row[i])

    csv_file.close()

    # Reading csv file for female data
    with open("../data/female_train_data.csv", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        femaleData = dict()
        header = next(reader)   
        for name in header:
            femaleData[name] = list()

        for row in reader:
            for i in range(len(row)):
                femaleData[header[i]].append(row[i])

    csv_file.close()


    print("MaleData 10th row, index %s, bmi %s, stature_mm %s"%(maleData['index'][9], maleData['male_bmi'][9], maleData['male_stature_mm'][9]))
    print("FemaleData 10th row, index %s, bmi %s, stature_mm %s"%(femaleData['index'][9], femaleData['female_bmi'][9], femaleData['female_stature_mm'][9]))
