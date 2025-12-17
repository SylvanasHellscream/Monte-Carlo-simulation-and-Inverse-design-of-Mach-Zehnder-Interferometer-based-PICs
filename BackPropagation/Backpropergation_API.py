import numpy as np
#import lumapi
import matplotlib.pyplot as plt
#import pandas  as pd
#from lumopt.optimizers.generic_optimizers import ScipyOptimizers
#from scipy.optimize import minimize

import importlib.util

#Change the version name or path if this path is not correct: C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v232\\api\\python\\lumapi.py')

#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) #windows
spec_win.loader.exec_module(lumapi)
print("Lumerical interface import done!")
print("You may get a warning, it is fine.")


It=lumapi.INTERCONNECT("BackPropergation.icp")
#Set up variables
Expect_output_0=np.array([0.67621919, 1.82835547, 0.71661304, 0.81740742, 1.25945667, 1.21441262, 1.14283372, 0.34256402])
initial_coupling_ratio=0.5 #Set initial coupling ratio to 0.45/0.55

Column=8 
Row=8  # Set up matric size

InputLaserVector=1e-3 * np.array([1,1,1,1,1,1,1,1]) #Set up input, sum matters
Expect_output=Expect_output_0*np.sum(InputLaserVector)/np.sum(Expect_output_0)*1000
print(Expect_output)
print(InputLaserVector)

x0=np.repeat(initial_coupling_ratio,(Row-1)*Column/2)
#x0=np.array([0.52108572, 0.47908802, 0.27219309, 1.1693688,  0.09585711, 0.32744368])

#define the sweep function for optimize
def sweep (x): #parameter x types array, contain all coupling ratio of elements
    #print(x) 
    x[x<0]=0
    x[x>1]=1
    #reset every element
    for i in range(1,Column//2+1):
        for j in range(1,Row):
            ElementName="C_"+str(i)+"_"+str(j)
            #print(ElementName)
            It.select(ElementName)
            It.set("coupling coefficient 1",x[j+(i-1)*(Row-1)-1])
        
    It.run()
    SweepResult=np.zeros(Row)
    for i in range (Row):
        MeterName="OPWM_"+str(i+1)
        #temp=getresultdata(MeterName,"sum/power");
        SweepResult[i]=It.getresultdata(MeterName,"sum/power")
    It.switchtodesign()
    SweepResult=10**(SweepResult/10)
    print(SweepResult)
    print(np.sum((SweepResult-Expect_output)**2))
    LossFunction.append(np.sum((SweepResult-Expect_output)**2))
    return np.sum((SweepResult-Expect_output)**2)



It.putv("Ncolumn",Column)
It.putv("Nrow",Row)
It.putv("InsersionLoss_Var",initial_coupling_ratio)


code = open("ConstructMatrix.lsf","r").read()
It.eval(code)
for i in range(Row):
    Laser="CWL_"+str(i+1)
    
    It.select(Laser)
    It.set("power",InputLaserVector[i])

'''res = minimize(sweep, x0, method='nelder-mead',
               options={'xatol': 1e-5, 'disp': True})'''#scipy.optimize.minimize is not effective
#Set up recording variables
LossFunction=[]


# CMA approach
import cma
es = cma.CMAEvolutionStrategy(x0, 0.1,{'maxiter': 200}).optimize(sweep)
res = es.result
print("Optimized x:", res.xbest)
plt.scatter(np.arange(len(LossFunction)),LossFunction)
plt.xlabel("Iteration")
plt.ylabel("progress loss function")
plt.show()

'''# Swarm particle 
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
optimizer = ps.single.GlobalBestPSO(n_particles=x0.size, dimensions=2, options='options')'''

'''# dual_annealing
from scipy.optimize import dual_annealing
bounds=[(0, 1) for _ in range(x0.size)]
result = dual_annealing(sweep, bounds=bounds,initial_temp=10000)
print("Optimal parameters:", result.x)
print("Optimal value:", result.fun)'''

'''# Generic Algorithm
import pygad
ga_instance = pygad.GA(num_generations=100, # Number of generations.
                       num_parents_mating=2, # Number of solutions to be selected as parents in the mating pool.
                       fitness_func=sweep,
                       sol_per_pop=10, # Number of solutions in the population.
                       num_genes=x0.size, # Number of genes in each solution.
                       init_range_low=0,
                       init_range_high=1,
                       mutation_percent_genes=10, # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
                       parent_selection_type="sss", # Method used to select parents. sss is steady-state selection.
                       crossover_type="single_point", # Type of the crossover operator.
                       mutation_type="random", # Type of the mutation operator.
                       keep_parents=1, # Number of parents to keep in the current population. -1 means keep all parents and 0 means keep nothing.
                       )
# Running the GA to optimize the parameters of the function.
ga_instance.run()
# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Solution: {solution}")
print(f"Best Solution Fitness: {-solution_fitness}") '''

