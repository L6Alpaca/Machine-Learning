import numpy as np
def softmax(z):
    return np.exp(z)/np.exp(z).sum()

z = [3,1,-3]
print(softmax(z).round(2))#round(n)保留n位小数