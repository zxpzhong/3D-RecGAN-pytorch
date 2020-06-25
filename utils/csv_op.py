import sys
import numpy as np
import os
import csv
import pandas as pd

def write_to_csv(filename,title,list_data):
    with open(filename, "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect=("excel"))
        csvwriter.writerow(title)
        for count,item in enumerate(list_data):
            csvwriter.writerow(item)

def read_from_csv(filename):

    data = pd.read_csv(filename)
    return data


if __name__ == '__main__':
    # write_to_csv('test.csv',['data1','data2','data3'],[[1,2,3],[4,5,6]])
    data = read_from_csv('test.csv')
    print(data)