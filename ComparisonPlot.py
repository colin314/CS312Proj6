import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv',delimiter=',')
nPts = [15,30,60,100]#,200]
itCount = [50,100,250]#,500,1000]

plt.figure()
for it in itCount:
    y = []
    for n in nPts:
        for i in range(len(data)):
            print(data[i,0],data[i,9])
            if not np.isnan(data[i,0]) and not np.isnan(data[i,9]):
                print(n,data[i,0],data[i,9])
                if int(data[i,0]) == n and int(data[i,9]) == it:
                    print('adding')
                    y.append(data[i,11])
                    break
    plt.plot(nPts,y,'.',label='Anneal - ' + str(it))
plt.legend()
plt.xticks(nPts)
plt.show()

    
