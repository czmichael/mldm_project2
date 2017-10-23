import numpy as np

x = np.full((3, 1), 2.0)
print(x)

y = np.full((1, 3), 3.0)
print(y)

print np.dot(y, x)



a1 = np.matrix("2; 2; 2")
b1 = np.matrix("3 3 3")
c1 = np.multiply(a1, b1)
print(c1)