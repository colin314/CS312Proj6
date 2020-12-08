import numpy as np
import matplotlib.pyplot as plt

x0 = 500
xmin = 0.1
def dec(x):
    return x - 0.01*x
x = [x0]

while x[-1] > xmin:
    x.append(dec(x[-1]))

xx = np.arange(0,len(x),1)

plt.plot(xx,x)
plt.show()