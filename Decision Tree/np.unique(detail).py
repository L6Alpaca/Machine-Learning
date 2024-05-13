import numpy as np
import math
dataSet = np.array([[0, 0, 0, 0, 'no'],
                    [0, 0, 0, 1, 'no'],
                    [0, 1, 0, 1, 'yes'],
                    [0, 1, 1, 0, 'yes'],
                    [0, 0, 0, 0, 'no'],
                    [1, 0, 0, 0, 'no'],
                    [1, 0, 0, 1, 'no'],
                    [1, 1, 1, 1, 'yes'],
                    [1, 0, 1, 2, 'yes'],
                    [1, 5, 1, 2, 'yes'],
                    [2, 0, 1, 2, 'yes'],
                    [2, 3, 1, 1, 'yes'],
                    [2, 1, 0, 1, 'yes'],
                    [2, 1, 0, 2, 'yes'],
                    [2, 0, 0, 0, 'no']])
unique = np.unique(dataSet[ : ,1])
print(unique)
print(0.4* math.log(0.4,2)+0.6*math.log(0.6,2))