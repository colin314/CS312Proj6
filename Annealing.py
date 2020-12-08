import numpy as np
import matplotlib.pyplot as plt

x0 = 500
xmin = 0.01
def dec(x):
    return x - 1/x
x = [x0]

while x[-1] > xmin:
    x.append(dec(x[-1]))

xx = np.arange(0,len(x),1)

plt.plot(xx,x)
plt.show()
x = np.array(x)
plt.plot(x,np.exp(-1/x))
plt.show()

Tmax = 500
Tmin = 0
nPts = 8000
Tx = np.linspace(0,np.pi*4,nPts)
def f_T(x):
    return Tmax*np.cos(x/4) + Tmax

plt.plot(Tx,f_T(Tx))
plt.show()