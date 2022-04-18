# Linear Algebra Learning Sequence
# randomisation_matrix() Function
# to generate Random Matrix

import numpy as np


# defining a function in input arguements
# as row and column numbers
def randomization_matrix(m,n):

    A = np.random.uniform(1.5000, 10.9000, size=(m, n))
    np.round(A,4,A)
    return A

def getFrobenius_Norm(A):
    return(np.linalg.norm(A, 'fro'))


if __name__ == "__main__":
    # Generating Randome Floats
    m = int(input('Number of Rows: '))
    n = int(input('Number of Columns: '))

    # printing the random matrix
    A = randomization_matrix(m, n)
    print("\n The generated Random {} / {} matrix is :\n{}".format(m,n,A))

    F = getFrobenius_Norm(A)
    print("\nFrobenius norm for the generate matrix is :")
    print(round(F,4))
