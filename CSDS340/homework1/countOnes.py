# Problem 2

''' Write a function that consists of a set of loops that run through a 2-D
    numPy array and counts the number of ones in it. Do the same thing using
    the np.where() function. Name your functions countOnesLoop() and 
    countOnesWhere(). 
''' 

def countOnesLoop(arr):
    count = 0
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        for row in arr:
            for element in row:
                if element == 1:
                    count += 1
    else:
        print("Not a 2-D array")
    return count

ones = np.full((6,4),1)
print(countOnesLoop(ones))

def countOnesWhere(arr):
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        indices = np.where(arr == 1)
        return len(indices[0])
    else:
        print("Not a 2-D array")

print(countOnesWhere(ones))