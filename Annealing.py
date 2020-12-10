import numpy as np
import matplotlib.pyplot as plt
import random
# x0 = 500
# xmin = 0.01
# def dec(x):
#     return x - 1/x
# x = [x0]

# while x[-1] > xmin:
#     x.append(dec(x[-1]))

# xx = np.arange(0,len(x),1)

# plt.plot(xx,x)
# plt.show()
# x = np.array(x)
# plt.plot(x,np.exp(-1/x))
# plt.show()

# Tmax = 500
# Tmin = 0
# nPts = 8000
# Tx = np.linspace(0,np.pi*4,nPts)
# def f_T(x):
#     return Tmax*np.cos(x/4) + Tmax

# plt.plot(Tx,f_T(Tx))
# plt.show()

# data = []
# imax = 500
# imin = 100
# width = 10
# ri = random.randint(0,500)
# for i in range(200):
#     start = max(imin,ri-width)
#     end = min(imax,ri+width)
#     ri = random.randint(start,end)
#     data.append(ri)
# data = np.array(data)
# np.savetxt('data.txt',data,delimiter=',')

data = np.genfromtxt('data.txt',delimiter=',')
sortedData = data.argsort()[-5:][::-1]
print(sortedData)
x = np.arange(0,200,1)
plt.plot(x,data)
plt.plot(sortedData[0])
for i in sortedData:
    plt.plot([i,i],[200,data[i]])
plt.ylim([200,350])
plt.show()