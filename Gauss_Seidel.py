# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix
import numpy as np
from numpy import linalg as LA
def seidel(a, xold, b):
    # Finding length of a(3)
    x = xold
    n = len(a)
    # for loop for 3 times as to calculate x, y , z
    for j in range(0, n):
        # temp variable d to store b[j]
        d = b[j]

        # to calculate respective xi, yi, zi
        for i in range(0, n):
            if (j != i):
                d -= a[j][i] * x[i]
        # updating the value of our solution
        x[j] = d / a[j][j]
    # returning our updated solution
    return x


# int(input())input as number of variable to be solved
n = 4
a = []
b = []
converged = False
# initial solution depending on n(here n=3)
xold = [0, 0, 0, 0]
a = np.random.randint(1,10, size =(4, 4))
print(f"The Input A Matrix is :: \n" + str(a))

#a = [[4, 1, 2], [3, 5, 1], [1, 1, 3]]
b = np.random.randint(1,10,size=4)
print(f"The Input b Matrix is :: \n"+ str(b))

print("Intial Solution Matrix is ::\n")
print(xold)

print("Below are the Iteration Matrices ::\n")
# loop run for m times depending on m the error value
for i in range(0, 100):
    xpass =[]
    xpass=np.append(xpass,xold)
    xnew = seidel(a, xpass, b)
    print(xnew)
    print("The Frobe-nius norms of the matrix is ::"+str(LA.norm(xnew))+".\nThe L-infinite norm of the matrix is ::"+str(LA.norm(xnew,np.inf))+".\nThe L-1 norm of the matrix is ::"+str(LA.norm(xnew,1))+"\n")
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


