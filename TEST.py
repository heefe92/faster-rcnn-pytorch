import numpy as np

a = np.ones((1000,1000,1000))
b = a[np.arange(0,10),np.array([0,10])]
print(b.shape)