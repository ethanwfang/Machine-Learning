# Problem 1
import numpy as np
# a) Make an array a of size 6x4 where every element is a 2
a = np.full((6,4), 2)
print(a)

# b) Make an array b of size 6x4 that has 3 on the leading diagonal and 1 everywhere else
b = np.ones((6,4))
np.fill_diagonal(b, 3,)
print(b)

# c) Can you multiply these two matrices together? Wy does a * b work, but not np.dot(a,b)
print(a*b)
'''
    The operation a*b seems to multiply the elements of the matrices together. This means
    that the element a(1,1) is multiplied by b(1,1) to optain element (1,1) in resulting matrix c, if c = a*b.
    This means that you can only perform this "*" operation on matrices of the exact same dimension. 

    The dot product between a and b do not work because the convolutions between the
    sizes of the matrices go out of bounds
'''

# d) Compute np.dot(a.transpose(), b) and np.dot(a,b.transpose()). Why are the results different shapes?
np.dot(a.transpose(), b) # 4x4 array filled with 16
np.dot(a, b.transpose()) # 6x6 array with 12 in first 4 columsn and 8 in last 2

'''
    The results are different shapes because a.transpose() dot b is a 4x6 dot 6x4 operation,
    which results in a 4x4 resulting matrix.

    The latter dot product is a 6x4 dot 4x6 matrix dot operation, which results in a 
    6x6 resulting matrix.
'''

print()