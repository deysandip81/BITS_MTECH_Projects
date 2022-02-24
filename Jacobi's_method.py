# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix
import numpy as np
from numpy import linalg as LA
import scipy.linalg as la
def normalize_matrix(a):
    x = np.random.uniform(0, 1, size=(4,4))

    #normalized_a = np.random.randint(0,1, size =(4, 4))
    for i in range(4):
        for j in range(4):
            x[i,j] = (a[i,j] / a[i,i])
    return x

def lu_decomp(x):
    I = np.identity(4)
    L = np.random.uniform(0, 0, size=(4,4))
    U = np.random.uniform(0, 0, size=(4,4))
    # print("Initial L matrix is ::"+ str(L))
    # print("Initial U matrix is ::"+ str(U))

    for i in range(4):
        for j in range(4):
            if i>j:
                L[i][j] = x[i][j]
            elif i<j:
                U[i][j]=x[i][j]
    return I,L,U

def jacobi_coefficient(I,L,U):
    # I =  np.identity(4)
    # P, L, U = la.lu(a)
    # print(np.dot(L, U))
    # print(P)
    # print(L)
    # print(U)
    return(-(np.add(L,U)))

def matrix_norms(jacobi_coeff):
    print("The Frobenius norms of the matrix is ::"+str(LA.norm(jacobi_coeff))+".\nThe L-infinite norm of the matrix is ::"+str(LA.norm(jacobi_coeff,np.inf))+".\nThe L-1 norm of the matrix is ::"+str(LA.norm(jacobi_coeff,1))+"\n")

# int(input())input as number of variable to be solved
n = 4
a = []
b = []
converged = False
# initial solution depending on n(here n=3)
xold = [0, 0, 0, 0]
a = np.random.randint(1,10, size =(4, 4))
print(f"The Input A Matrix is :: \n" + str(a))

x = normalize_matrix(a)

print("The normalized Matrix is ::\n" + str(x))


I,L,U = lu_decomp(x)

print("The identity matrix is ::\n"+ str(I))
print("The Lower matrix is ::\n"+ str(L))
print("The Upper matrix is ::\n"+ str(U))

jacobi_coeff = np.random.uniform(0, 0, size=(4,4))
jacobi_coeff = jacobi_coefficient(I,L,U)
print("The Coefficient Array is ::\n"+ str(jacobi_coeff))
matrix_norms(jacobi_coeff)


