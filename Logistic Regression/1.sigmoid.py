import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.linspace(-5,5,100)
#从-5到5均等划分100份 float
print(sigmoid(x))
#sigmoid图像y属于(-1,1)
