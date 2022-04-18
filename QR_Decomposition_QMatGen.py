import numpy as np
from sympy import Matrix, init_printing
from math import sqrt, floor
import scipy
import scipy.linalg
init_printing()

def randomization_matrix(m,n,lb,ub):

    A = np.random.uniform(lb, ub, size=(m, n))
    np.round(A,4,A)
    return A

def getFrobenius_Norm(A):
    return floor(np.linalg.norm(A, 'fro') * 10**2)/ 10**2
    #return np.linalg.norm(A, 'fro')

def norm(x):
    """Return the Euclidean norm of the vector x."""
    return sqrt(sum([x_i**2 for x_i in x]))


def getRank(A):
    return np.linalg.matrix_rank(A)

def cmp(a, b):
    return bool(a > b) - bool(a < b)

def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = zip(*N)

    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

def trans_matrix(M):
    """Take the transpose of a matrix."""
    n = len(M)
    return [[ M[i][j] for i in range(n)] for j in range(n)]

def Q_i(Q_min, i, j, k):
    """Construct the Q_t matrix by left-top padding the matrix Q
    with elements from the identity matrix."""
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]


def gramschmidt_Q(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q matrix only.
    """
    """Performs a Householder Reflections based QR Decomposition of the                                               
    matrix A. The function returns Q, an orthogonal matrix ."""
    n = len(A)
    n1 = min(len(A), len(A[0]))
    print("The Value of n is {}".format(n))
    # Set R equal to A, and create Q as a zero matrix of the same size
    R = A
    Q = [[0.0] * n for i in range(n)]

    # The Householder procedure
    for k in range(n1-1):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
        # Create identity matrix of same size as A
        I = [[float(i == j) for i in range(n)] for j in range(n)]

        # Create the vectors x, e and the scalar alpha
        # Python does not have a sgn function, so we use cmp instead
        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -cmp(x[0],0) * norm(x)

        # Using anonymous functions, we create u and v
        u = list(map(lambda p,q: p + alpha * q, x, e))
        norm_u = norm(u)
        v = list(map(lambda p: p/norm_u, u))

        # Create the Q minor matrix
        Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in range(n-k)] for j in range(n-k) ]

        # "Pad out" the Q minor matrix with elements from the identity
        Q_t = [[ Q_i(Q_min,i,j,k) for i in range(n)] for j in range(n)]

        # If this is the first run through, right multiply by A,
        # else right multiply by Q
        if k == 0:
            Q = Q_t
            #R = mult_matrix(Q_t,A)
            Q = np.around(Q,4)
        else:
            Q = np.matmul(Q_t,Q)
            Q = np.around(Q,4)

    # Since Q is defined as the product of transposes of Q_t,
    # we need to take the transpose upon returning it
    return np.asmatrix(trans_matrix(Q))

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

    while (1):
        if  rank == min(m,n) :
            print("The matrix is having Full Rank ( Rank = {} ). so it can be decomposed using Gram Schmidt method".format(rank))
            Q = gramschmidt_Q(A)
            print("Q Matrix is  = \n")
            print(Q)
            break
        else:
            print("The rank of the matrix is {}. As it is < {} , it does not have a Full Rank and hence can not be decomposed by Gram Schmidt method". format(rank, min(m,n)))


