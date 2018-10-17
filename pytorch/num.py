import numpy as np
num = np.zeros([2, 3])
[rows, cols] = num.shape
print(rows, cols)
for i in range(rows):
    for j in range(cols):
        print(num[i, j])
