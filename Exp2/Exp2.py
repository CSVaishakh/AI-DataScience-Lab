def readMat():
    rows = int(input('Enter the no of rows'))
    col = int(input('Enter the no of col'))
    
    mat=[]
    for i in range(rows):
        row=[]
        for i in range(col):
            ele =  int(input("Enter the element"))
            row.append(ele)
        mat.append(row)

    return mat


matA = readMat()
matB = readMat()

import numpy as np

matA = np.array(matA)
matB = np.array(matB)

print(f"matA : \n {matA} \n")
print(f"matB : \n {matB} \n")

mat_sum = matA + matB
print(f"sum : \n {mat_sum} \n")

diff = matA - matB
print(f"difference : \n {diff} \n")

prod = matA @ matB
print(f"product : \n {prod} \n")
