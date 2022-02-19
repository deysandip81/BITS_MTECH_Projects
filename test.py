import numpy as np
# defining a function for
# round s.
def round_s(n, decimals=0):
    if decimals == 0:
        n_5s = n + 5 * pow(10, -(decimals+1))
    else:
        n_5s = n + 5 * pow(10, -(decimals))
    if '.' in str(n_5s) and '-' in str(n_5s):
        return(float(str(n_5s)[0:decimals+2]))
    elif '.' in str(n_5s) or '-' in str(n_5s):
        return (float(str(n_5s)[0:decimals + 1]))
    else:
        return (float(str(n_5s)[0:decimals]))


def Cal_LU(D,g,ds):
    A=np.array((D),dtype=float)
    f=np.array((g),dtype=float)
    num_add = 0
    num_sub = 0
    num_div = 0
    num_mult = 0
    n = f.size
    for i in range(0,n-1):     # Loop through the columns of the matrix
        for j in range(i+1,n):     # Loop through rows below diagonal for each column
            if A[i,i] == 0:
                print("Error: Zero on diagonal!")
                print("Need algorithm with pivoting")
                break
            m = round_s(float(A[j,i]/A[i,i]),ds)
            num_div += 1
            A[j,:] = (A[j,:] - m*A[i,:])
            num_mult += n
            num_sub += n
            f[j] = round_s(float(f[j] - m*f[i]),ds)
            num_mult += 1
            num_sub += 1
    return A,f,num_add,num_sub,num_div,num_mult


def Cal_LU_pivot(D, g,ds):
    A = np.array((D), dtype=float)
    f = np.array((g), dtype=float)
    num_add = 0
    num_sub = 0
    num_div = 0
    num_mult = 0
    num_swaps = 0
    n = len(f)
    for i in range(0, n - 1):  # Loop through the columns of the matrix

        if np.abs(A[i, i]) == 0:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]] = A[[k, i]]  # Swaps ith and kth rows to each other
                    f[[i, k]] = f[[k, i]]
                    num_swaps += 2
                    break

        for j in range(i + 1, n):  # Loop through rows below diagonal for each column
            m = round_s(float(A[j,i]/A[i,i]),ds)
            num_div += 1
            A[j,:] = (A[j,:] - m*A[i,:])
            num_mult += n
            num_sub += n
            f[j] = round_s(float(f[j] - m*f[i]),ds)
            num_mult += 1
            num_sub += 1

    print("Total Number of swaps done ::" + str(num_swaps))
    return A, f,num_add,num_sub,num_div,num_mult

def Back_Subs(A, f,ds,num_add,num_sub,num_div,num_mult):
    n = f.size
    x = np.zeros(n)  # Initialize the solution vector, x, to zero
    if A[n - 1, n - 1] != 0:
        x[n - 1] = round_s(float(f[n - 1] / A[n - 1, n - 1]),ds)  # Solve for last entry first
        num_div += 1
    else:
        x[n - 1] = (f[n - 1])
    #x[n - 1] = (f[n - 1] / A[n - 1, n - 1])  # Solve for last entry first
    for i in range(n - 2, -1, -1):  # Loop from the end to the beginning
        sum_ = 0
        for j in range(i + 1, n):  # For known x values, sum and move to rhs
            sum_ = round_s(float(sum_ + A[i, j] * x[j]),ds)
            num_add += 1
            num_mult += 1
        x[i] = round_s(float((f[i] - sum_) / A[i, i]),ds)
        num_sub += 1
        num_div +=1
    return x,num_add,num_sub,num_div,num_mult

def gauss_without_pivot(A,f,ds):

    B,g,num_add,num_sub,num_div,num_mult = Cal_LU(A,f,ds)
    x ,num_add_final,num_sub_final,num_div_final,num_mult_final= Back_Subs(B, g,ds,num_add,num_sub,num_div,num_mult)
    print("\n\t The solution Vector is "+ str(x) +"\n\n")
    print(" Total Number additions :: " + str(num_add_final))
    print(" Total Number subtraction :: " + str(num_sub_final))
    print(" Total Number Divisions :: " + str(num_div_final))
    print(" Total Number Multiplication :: " + str(num_mult_final))
def gauss_with_pivot(A,f,ds):

    B, g,num_add,num_sub,num_div,num_mult = Cal_LU_pivot(A,f,ds)
    x ,num_add_final,num_sub_final,num_div_final,num_mult_final= Back_Subs(B, g,ds,num_add,num_sub,num_div,num_mult)
    print("\n\t The solution Vector is "+ str(x) +"\n\n")
    print(" Total Number additions :: " + str(num_add_final))
    print(" Total Number subtraction :: " + str(num_sub_final))
    print(" Total Number Divisions :: " + str(num_div_final))
    print(" Total Number Multiplication :: " + str(num_mult_final))

def main():
    # print(round_s(1.23454621, 0))
    # Reading number of unknowns
    n = int(input('Enter number of unknowns: '))

    # Making numpy array of n x n+1 size and initializing
    # to zero for storing augmented matrix
    A = np.zeros((n, n))

    # Making numpy array of n size and initializing
    # to zero for storing value vector
    f = np.zeros(n)

    print('Enter Matrix Coefficients:')
    for i in range(n):
        for j in range(n):
            A[i][j] = float(input('a[' + str(i) + '][' + str(j) + ']='))

    print('Enter Value Matrix Coefficients:')
    for i in range(n):
        f[i] = float(input('f[' + str(i) + ']='))
        print(f)
    ds = int(input("Enter the decimal point correction you want:"))
    print("\n Solving Using Gauss Elimination without pivot: :")
    gauss_without_pivot(A,f,ds)
    print("\nnSolving Using Gauss Elimination with pivot: :")
    gauss_with_pivot(A,f,ds)




if __name__ == "__main__":
    main()

