#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))



import matplotlib.pyplot as plt
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
        #Initialization
        bestPath = []
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        k = 0
        start_time = time.time()
        #Create cost matrix
        count = 0
        foundRoute = False
        while not foundRoute and time.time() - start_time < time_allowance:
            costMatrix = np.full((ncities,ncities), np.inf)
            for i in range(len(cities)):
                for j in range(len(cities)):
                    costMatrix[i,j] = cities[i].costTo(cities[j])
            costMatrix[:,0] = np.inf
            for i in range(count):
                j = costMatrix[0].argmin()
                costMatrix[0,j] = np.inf
            sol = [0]
            i = 0
            while len(sol) < ncities:
                j = costMatrix[i].argmin()
                sol.append(j)
                costMatrix[i] = np.inf
                costMatrix[:,j] = np.inf
                costMatrix[j,i] = np.inf
                i = j
            count += 1
            sol = [cities[i] for i in sol]
            sol = TSPSolution(sol)
            if sol.cost < np.inf:
                foundRoute = True
            
        end_time = time.time()
        results = {}
        bssf = sol
        results['cost'] = bssf.cost if foundRoute else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results
    
    def greedyNeighbor(self, cities, start_time, time_allowance, oldCities, start, end):
        #Initialization
        bestPath = []
        results = {}
        ncities = len(cities)
        k = 0

        #Create cost matrix
        count = 0
        foundRoute = False
        while not foundRoute and time.time() - start_time < time_allowance:
            costMatrix = np.full((ncities,ncities), np.inf)
            for i in range(len(cities)):
                for j in range(len(cities)):
                    costMatrix[i,j] = cities[i].costTo(cities[j])
            costMatrix[:,0] = np.inf
            for i in range(count):
                j = random.randint(1,len(cities)-1)
                j = costMatrix[0].argmin()
                costMatrix[0,j] = np.inf
            sol = [0]
            i = 0
            while len(sol) < ncities:
                j = 0
                while costMatrix[i,j] == np.inf and j < len(cities):
                    j += 1
                if j >= len(cities):
                    break
                sol.append(j)
                costMatrix[i] = np.inf
                costMatrix[:,j] = np.inf
                costMatrix[j,i] = np.inf
                i = j
            count += 1
            sol = oldCities[:start] + [cities[i] for i in sol] + oldCities[(end+1):]
            sol = TSPSolution(sol)
            if sol.cost < np.inf:
                foundRoute = True
            
        return sol

    def reduceCost(self, cost, totCost):
        for i in range(len(cost)):
            minVal = cost[i,:][cost[i,:].argmin()]
            if minVal > 0 and minVal != np.inf:
                cost[i,:] -= minVal
                totCost += minVal
        for i in range(len(cost)):
            minVal = cost[:,i][cost[:,i].argmin()]
            if minVal > 0 and minVal != np.inf:
                cost[:,i] -= minVal
                totCost += minVal
        return totCost
    
    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
        
    def branchAndBound( self, time_allowance=60.0 ):
        # Prepare statistic variables
        childStates = 0
        pruned = 0
        bssfUpdates = 0

        # initialize BSSF by using random tour
        randomSol = self.defaultRandomTour()
        BSSF = randomSol['cost']

        #Initialization
        bestPath = []
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        k = 0
        
        #Create cost matrix
        costMatrix = np.full((ncities,ncities), np.inf)
        for i in range(len(cities)):
            for j in range(len(cities)):
                costMatrix[i,j] = cities[i].costTo(cities[j])

        #Time Start
        start_time = time.time()
        
        #Reduce initial matrix
        cost_i = self.reduceCost(costMatrix,0)


        # inf_mask = np.ma.masked_equal(costMatrix, np.inf, True)
        # avgCost = np.ma.average(inf_mask)
        
        #Initialize priority queue
        queue = []
        heapq.heappush(queue,(ncities-1,cost_i,0,1, k, costMatrix,[0]))
        childStates += 1
        k += 1
        maxQueueLen = len(queue)

        #Begin branch and bound
        while len(queue) > 0 and time.time() - start_time < time_allowance:
            maxQueueLen = len(queue) if len(queue) > maxQueueLen else maxQueueLen

            #pop off queue
            key, cost, i, visitCount, z, costMatrix, path = heapq.heappop(queue)

            #If BSSF is updated states on queue may have become stale
            if cost > BSSF:
                pruned += 1
                continue

            #Loop over all the cities
            for j in range(ncities):
                #Skip if city j is unreachable from city i
                if costMatrix[i,j] != np.inf:
                    childStates += 1
                    pathCost = cost + costMatrix[i,j] #Cost to go from i to j
                    #Prune if cost is > BSSF
                    if pathCost < BSSF:
                        #Now reduce the cost matrix set row i, column j, and [j,i] to inf
                        reducedCostMatrix = np.copy(costMatrix)
                        reducedCostMatrix[i,:] = np.inf
                        reducedCostMatrix[:,j] = np.inf
                        reducedCostMatrix[j,i] = np.inf
                        reducedCost = self.reduceCost(reducedCostMatrix,pathCost)
                        if reducedCost < BSSF:
                            newPath = path.copy()
                            newPath.append(j)
                            if visitCount + 1 == ncities:
                                BSSF = reducedCost
                                bestPath = newPath
                                bssfUpdates += 1
                            else:
                                heapq.heappush(queue,(ncities-(visitCount + 1),reducedCost, j, visitCount + 1, k, reducedCostMatrix, newPath))
                                k += 1
                        else:
                            pruned += 1
                    else:
                        pruned += 1
                                
        end_time = time.time()
        print(bestPath)
        for i in range(len(bestPath)-1):
            print(f'{cities[bestPath[i]]._name} to {cities[bestPath[i+1]]._name} = {cities[bestPath[i]].costTo(cities[bestPath[i+1]])}')
        print(f'{cities[-1]._name} to {cities[0]._name} = {cities[-1].costTo(cities[0])}')
        path = [cities[i] for i in bestPath]
        results['cost'] = BSSF
        results['time'] = end_time - start_time
        results['count'] = bssfUpdates
        results['soln'] = TSPSolution(path)
        results['max'] = maxQueueLen
        results['total'] = childStates
        results['pruned'] = pruned
        return results



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''
        
    def fancy( self,time_allowance=60.0 ):
        cities = self._scenario.getCities()
        ncities = len(cities)
        # Tuning Parameters
        Tmax = 500
        Tmin = 0
        nPts = 1000*ncities
        Tx = np.linspace(0,Tmax,nPts)
        def f_T(x):
            return 1/(1+np.exp(x/100))*Tmax*2
        results = {}
        sol_y = []
        def randCost(dE, T): #FIXME: Tune function
            c_corrected = np.exp(-dE/T)
            p = 1/500**2*(T)**2
            r = rand()
            if c_corrected > r:
                #print(T,c_corrected, r,p)
                return True
            else:
                return False

        count = 0
        start_time = time.time()
        C = self.getInitialSol() 
        T, Tmin, dec_T = self.getT() 
        for T in Tx:
            if time.time()-start_time > time_allowance:
                break
            if count > 1000:
                break
            T = f_T(T)
            count += 1
            C_n = self.getNeighbor(C, start_time, time_allowance) 
            if C_n.cost != np.inf:
                delta_cost = C_n.cost - C.cost
                if delta_cost < 0:
                    C = C_n
                    count = 0
                elif randCost(delta_cost,T): 
                    C = C_n
                    count = 0
            sol_y.append(C.cost)
        plt.plot(np.arange(0,len(sol_y),1),sol_y)
        plt.savefig('Anneal.png')
        end_time = time.time()
        results = {}
        results['cost'] = C.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = C
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results
        
    def getNeighbor(self, C, start_time, time_allowance):         
        cities = C.route
        start = random.randint(0,len(cities) - 2)
        end = random.randint(start + 1, len(cities) - 1)
        C_n = self.greedyNeighbor(cities[start:end + 1], start_time, time_allowance, C.route, start, end)
        return C_n


    def getT(self):
        #(T, Tmin, dec_T)
        Tmax = 500
        Tmin = 0.1
        def dec_T(T):
            return T - 0.001*T
        return (Tmax, Tmin, dec_T)

    def getInitialSol(self):
        sol = self.greedy()
        return sol['soln']


