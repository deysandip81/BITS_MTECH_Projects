# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix
import numpy as np
from numpy import linalg as LA
def jacobi(a, xold, b):
    # Finding length of a(3)
    x = xold
    n = len(a)
    D = np.diag(a)
    R = a-np.diagflat(D)
    x = (b - np.dot(R, x)) / D
    return x


# int(input())input as number of variable to be solved
n = 4
a = []
b = []
converged = False
# initial solution depending on n(here n=3)
xold = [0, 0, 0, 0]
#a = np.random.randint(1,2, size =(4, 4))
print(f"The Input A Matrix is :: \n" + str(a))

a = [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],[0, -3, -1, 8]]
print(f"The Input A Matrix is :: \n" + str(a))

b = [6,25,-11,15]
#b = np.random.randint(1,2,size=4)
print(f"The Input b Matrix is :: \n"+ str(b))

print("Intial Solution Matrix is ::\n")
print(xold)

print("Below are the Iteration Matrices ::\n")
# loop run for m times depending on m the error value
for i in range(0, 25):
    xpass =[]
    xpass=np.append(xpass,xold)
    xnew = jacobi(a, xpass, b)
    print(xnew)
    print("The Frobenius norms of the matrix is ::"+str(LA.norm(xnew))+".\nThe L-infinite norm of the matrix is ::"+str(LA.norm(xnew,np.inf))+".\nThe L-1 norm of the matrix is ::"+str(LA.norm(xnew,1))+"\n")
    # print each time the updated solution]
    if (abs(xnew[0] - xold[0])<=0.1) and (abs(xnew[1] - xold[1])<=0.1) and (abs(xnew[2] - xold[2])<=0.1) and (abs(xnew[3] - xold[3])<=0.1):
        converged = True
        break;
    else:
        xold = xnew

if converged :
    print(f"The system has converged after {i+1} iterations \n")
    print("Solution Matrix is :: \n" + str(xnew))
else:
    print(f"The system has not converged after {i+1} iterations. Increase iteration numbers \n")


