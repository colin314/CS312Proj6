from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import math
import random
import signal
import sys
import time
# Import in the code with the actual implementation
from TSPSolver import *
#from TSPSolver_complete import *
from TSPClasses import *
from matplotlib import pyplot as plt

class Proj5GUI:

    def __init__( self ):
        super(Proj5GUI,self).__init__()
        self._MAX_SEED = 1000 
        self.solver = TSPSolver( 'view' )
        SCALE = 1.0
        self.data_range		= { 'x':[-1.5*SCALE,1.5*SCALE], \
								'y':[-SCALE,SCALE] }

    def newPoints(self, nPts, seed):        
        self.seed = seed
        random.seed( seed )

        ptlist = []
        xr = self.data_range['x']
        yr = self.data_range['y']
        while len(ptlist) < nPts:
            x = random.uniform(0.0,1.0)
            y = random.uniform(0.0,1.0)
            if True:
                xval = xr[0] + (xr[1]-xr[0])*x
                yval = yr[0] + (yr[1]-yr[0])*y
                ptlist.append( QPointF(xval,yval) )
        return ptlist

    def generateNetwork(self, nPts, rand_seed=20):
        points = self.newPoints(nPts, rand_seed) # uses current rand seed
        diff = 'Hard'
        self._scenario = Scenario( city_locations=points, difficulty=diff, rand_seed=rand_seed )

    def solveClicked(self, nPts=15, itCount=100, sVal=100, f_Tmax=lambda n,i: 5*n,seed=20):                                # need to reset display??? and say "processing..." at bottom???
        seed = 20
        self.generateNetwork(nPts, seed)
        self.solver.setupWithScenario(self._scenario)

        max_time = 3600
        #greedy
        result = self.solver.greedy(max_time)
        greedyCost = result['cost']
        
        #annealed
        #result = self.solver.fancy(max_time,itCount,sVal,f_Tmax=f_Tmax)
        annealCost = 0#result['cost']
        annealTime = 0#result['time']
        
        # sol_y = result['sol_y']
        # sol_x = np.linspace(0,annealTime,len(sol_y))
        # plt.clf()
        # plt.plot(sol_x, sol_y,'k.',label=str(annealCost))
        # plt.title(r'$n_{pts} = ' + f'{nPts:.0f}$; ' + r'$n_{it} = ' + f'{itCount}$')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Cost')
        # plt.legend()
        # plt.savefig(f'Annealing_{nPts}Pts_{itCount}it.png',pad_inches=0.1,bbox_inches='tight')
        
        result = self.solver.defaultRandomTour(10)
        randTime = result['time']
        randCost = result['cost']
        result = self.solver.branchAndBound(600)
        BBTime =  result['time']
        BBCost = result['cost']
        
        row = np.array([nPts, round(randTime,2), int(randCost), int(greedyCost), round(greedyCost/randCost * 100,2), \
            round(BBTime,2), int(BBCost), round(BBCost/greedyCost*100,2), round(annealTime,2), itCount, int(annealCost), \
                round(annealCost/greedyCost*100,2)])
        csvRow = ",".join(row.astype(str))

        with open("data.csv", "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write('\n')
            file_object.write(csvRow)

gui = Proj5GUI()
nPts = [50,60,100,150,200]
itCount = [100]
sVals = [50,100,150,200,250]
n = 5


for n_p in nPts:
    try:
        gui.solveClicked(n_p,100,random.randint(0,1000))
    except:
        print('Whoops')


