# Python Program to check
# whether given matrix is
# Diagonally Dominant Matrix.

# check the given given
# matrix is Diagonally
# Dominant Matrix or not.
def isDDM(m, n):
    # for each row
    for i in range(0, n):

        # for each column, finding
        # sum of each row.
        sum = 0
        for j in range(0, n):
            sum = sum + abs(m[i][j])

            # removing the
        # diagonal element.
        sum = sum - abs(m[i][i])

        # checking if diagonal
        # element is less than
        # sum of non-diagonal
        # element.
        if (abs(m[i][i]) < sum):
            return False

    return True

# Function to return the minimum steps
# required to convert the given matrix
# to a Diagonally Dominant Matrix
def findStepsForDDM(arr,N):
    result = 0

    # For each row
    for i in range(N):

        # To store the sum of the current row
        sum = 0
        for j in range(N):
            sum += abs(arr[i][j])

        # Remove the element of the current row
        # which lies on the main diagonal
        sum -= abs(arr[i][i])

        # Checking if the diagonal element is less
        # than the sum of non-diagonal element
        # then add their difference to the result
        if (abs(arr[i][i]) < abs(sum)):
            result += abs(abs(arr[i][i]) - abs(sum))

    return result

# Driver Code
n = 3
m = [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]

if ((isDDM(m, n))):
    print("The matrix is diagonally dominant")
else:
    res = findStepsForDDM(m,n)
    if (res > 0):
        print(f"The matrix is not diagonally dominant but it can be made diagonally dominant by {res} steps ")
    else:
        print("The matrix is not diagonally dominant and  it can not be made diagonally dominant by row interchanges ")


