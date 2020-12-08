#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from random import random as rand
import copy


class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''
    
    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results


    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy( self,time_allowance=60.0 ):
        pass
    
    
    
    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
        
    def branchAndBound( self, time_allowance=60.0 ):
        pass



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''
        
    def fancy( self,time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        
        #DEBUG VALUES#
        randCostTrue = 0
        randCostTrueAfterHalfTempDecrease = 0
        randTrueAfterQuarterLeft = 0
        totalIter = 0
        ###
        
        start_time = time.time()
        C = self.getInitialSol() #FIXME: Get initial solution
        T, Tmin, dec_T = self.getT() #FIXME: Get T and T decrement function
        init_T = T
        while T > Tmin:
            C_n, iterations = self.getNeighbor(C) #FIXME: Implement
            print("TIMES IT TOOK TO FIND NEIGHBOR: " + str(iterations))
            totalIter += iterations
            if C_n.cost != np.inf:
                delta_cost = C_n.cost - C.cost
                if delta_cost < 0:
                    C = C_n
                elif self.randCost(delta_cost,T): 
                    randCostTrue += 1
                    
                    if (T < init_T/2): #start debug value counting for updates past halfway done
                        randCostTrueAfterHalfTempDecrease += 1
                        
                    if (T < init_T/4):
                        randTrueAfterQuarterLeft += 1
                    
                    C = C_n
            T -= dec_T
            
        end_time = time.time()
        
        print("AVERAGE ITER TO FIND NEIGHBOR = " + str(totalIter/(T/dec_T)))
        print("TIMES PROBABILITY WON: " + str(randCostTrue))
        print("TIMES PROBABILITY WON (MORE THAN HALFWAY THRU): " + str(randCostTrueAfterHalfTempDecrease))
        print("TIMES PROBABILITY WON (Last QUARTER): " + str(randTrueAfterQuarterLeft))

    

        pass
        
    def getNeighbor(self, C):
        for i in range(len(C.route)):
            print(C.route[i]._index, end="")
        print("\n")
        
        iterations = 0
            
        foundNeighbor = False
        while not foundNeighbor:
            iterations += 1
            randPlaceToBeginSwap = random.randint(0, len(C.route) - 2)
            randPlaceToEndSwap = random.randint(randPlaceToBeginSwap + 1, len(C.route) - 1)
            
            pathToAlter = copy.copy(C.route)
            
            slicePath = pathToAlter[randPlaceToBeginSwap:randPlaceToEndSwap + 1]
            for i in slicePath:
                print(i._index, end="")
            print("\n")
            
            pathToAlter[randPlaceToBeginSwap:randPlaceToEndSwap + 1] = reversed(pathToAlter[randPlaceToBeginSwap:randPlaceToEndSwap + 1])
            
            possibleNeighbor = TSPSolution(pathToAlter)
            
            for i in range(len(possibleNeighbor.route)):
                print(possibleNeighbor.route[i]._index, end="")
            print("\n")
            
            if (possibleNeighbor.cost != np.inf):
                foundNeighbor = True
                
        return possibleNeighbor, iterations
    
    #8 cities, 5000/5 Ts, 2.774 average iterations to find neighbor
    #10 cities, 5000/50 Ts, 3.8 average iter to find neighbor
    #10 cities, 10000/100 Ts, 2.94 average iter
    #15 cities, 5000/50 Ts, 5.26 average iter to find neighbor
    #15 cities, 10000/100 Ts, 3.87 average iter
    #15 cities, 50000/500 Ts, 4.17 average iter

    def getT(self):
        return 10000, 1, 100
    
    
    #FINDINGS/NOTES:
    #The smaller the initial T, the less ability it will have to "jump" localities (smaller denominators)
    #(Obvious) the larger diff between T and Tmin the more iterations
    #The larger the dec_T the more we can reduce iterations and maintain high Ts still
    
    def randCost(self, dE, T): #FIXME: Tune function
        c_corrected = np.exp(-dE/T)
        randNum = random.random()
        print("C_CORRECTED = " + str(c_corrected))
        return c_corrected > randNum #FIXME: Figure out random range
    
    #15 cities, 5000/50 Ts, 36 times randCost eval to True
    #15 cities, 1000/10 Ts, 19 times randCost eval to True
    #10 cities, 10000/100 Ts, 41 times eval to True
    #15 cities, 10000/100 Ts, 37 times randCost eval to True
    #15 cities, 10000/100 Ts, 43 times eval to True
    
    #15 cities, 10000/100 Ts, 41 times eval to True, 18 after 50% completion
    #15 cities, 50000/500 Ts, 48 times eval to True, 21 after 50% completion
    
    #15 cities, 1000/10 Ts, (15,23) times eval to True, (1, 10) after half, (0, 4) with 1/4 left
    #15 cities, 5000/50 Ts, (35, 28) times eval to True, (14, 11) after half, (5, 4) with 1/4 left
    #15 cities, 10000/100 Ts, (38 40) times eval to True, (16, 21) after half, (7,10) with 1/4 left  ###TOO HIGH
    #15 cities, 50000/500 Ts, 51 times eval to True, 26 after half, 12 with 1/4 left ###TOO HIGH
    
    
    ###TAKEAWAYS### With  higher T and dec_T values the randCost function won't actually restrict values
    #as the function gets closer to finishing
    
    



    def getInitialSol(self):
        #returns TSPSolution
        return self.defaultRandomTour()["soln"]


