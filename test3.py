import numpy as np
from sympy import Matrix, init_printing
init_printing()

def randomization_matrix(m,n,lb,ub):

    A = np.random.randint(lb, ub, size=(m, n))
    np.round(A,4,A)
    return A

def getRank(A):
    return np.linalg.matrix_rank(A)

if __name__ == "__main__":
    # Generating Randome Floats
    m = int(input('Number of Rows: '))
    n = int(input('Number of Columns: '))
    lb = float(input("Enter the lower bound of random array value:"))
    ub = float(input("Enter the upper bound of random array value:"))

    # printing the random matrix
    A = randomization_matrix(m, n, lb,ub)
    print("\n The generated Random {} / {} matrix is :\n{}".format(m,n,A))

    rank = getRank(A)

    if  rank == min(m,n) :
        print("The matrix is having Full Rank ( Rank = {} ). so it can be decomposed using Gram Schmidt method".format(rank))
    else:
        print("The rank of the matrix is {}. As it is < {} , it does not have a Full Rank and hence can not be decomposed by Gram Schmidt method". format(rank, min(m,n)))
