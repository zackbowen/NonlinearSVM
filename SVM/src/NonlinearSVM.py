import pandas as pd
import numpy as np
import csv

def main():
    #f = open("utils/iris_data", "r")
    #print(f.read())

    r = np.empty([1,5])
    r_float = r = np.empty([1,4])
    total= np.empty([150,5])
    i = 0;
    with open('utils/iris_data', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            r = np.array(row)
            # Convert the rows to floats
            r_float = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]

            total[i] =  [np.append(r_float, 1)]
            #if(i < 50):
            #    print(row)
            #elif(i < 100):
            #    print(row)
            #else:
            #    print(row)
            i+=1
        print(total)

if __name__ == "__main__":
    main()