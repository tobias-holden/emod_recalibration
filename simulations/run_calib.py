import os, sys, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval, GreaterThan, LessThan

sys.path.append("../")
from batch_generators.expected_improvement import ExpectedImprovement
from batch_generators.turbo_thompson_sampling import TurboThompsonSampling
from batch_generators.batch_generator_array import BatchGeneratorArray

from emulators.GP import ExactGP
from bo import BO
from plot import *

from my_func import my_func as myFunc
from compare_to_data.run_full_comparison import plot_all_comparisons
from clean_all import clean_analyzers
from translate_parameters import translate_parameters

# Define the square function in one dimension
#def myFunc(x):
#    return x**2 

# Define the Problem, it must be a functor
class Problem:
    def __init__(self,workdir="checkpoints/emod"):
        self.dim = 15 # mandatory dimension
        self.ymax = None #max value
        self.best = None
        self.n = 0
        self.workdir = workdir
        
        try:
            self.ymax = np.loadtxt(f"{self.workdir}/emod.ymax.txt").astype(float)
            self.n = np.loadtxt(f"{self.workdir}/emod.n.txt").astype(int)
            #self.best = np.loadtxt(f"{self.workdir}/emod.best.txt").astype(float)
        except IOError:
            self.ymax = None
            self.n = 0
            #self.best = None
            
                        
        os.makedirs(os.path.relpath(f'{self.workdir}/'), exist_ok=True)

    # The input is a vector that contains multiple set of parameters to be evaluated
    def __call__(self, X):
        # Each set of parameter x is evaluated
        # Note that parameters are samples from the unit cube in Botorch
        # Here we map unnormalizing them before calling the square function
        # Finally, because we want to minimize the function, we negate the return value
        # Y = [-myFunc(x) for x in unnormalize(X, [-5, 5])]
        # We return both X and Y, this allows us to disard points if so we choose
        # To remove a set of parameters, we would remove it from both X and Y

        # Finally, we need to return each y as a one-dimensional tensor (since we have just one dimension)
        # 
        #rewrite myfunc as class so we can keep track of things like the max value - aurelien does plotting each time but only saves when the new max > old max - would also allow for easier saving of outputs if desired. would also potentially help with adding iterations to param_set number so we don't reset each time. not sure yet if better to leave existing myfunc or pull everything into this
        param_key=pd.read_csv("test_parameter_key.csv")
        Y=myFunc(X)
        params=Y['param_set']
        Y=Y['ll']
        xc = []
        yc = []
        pc = []
        for j in range(len(Y)):
            if pd.isna(Y[j]):
                continue
            else:
                xc.append(X[j].tolist())
                yc.append([Y[j]])
                pc.append(params[j])
                
        xc2=[tuple(i) for i in xc]
        links=dict(zip(xc2,yc)) 
        pset=dict(zip(pc,yc))
        print(max(links.values())[0])
        print(self.ymax)
        print(max(pset,key=pset.get))
        # If new best value is found, save it and some other data
        if self.ymax is None:
            self.ymax = max(links.values())
            best_x = max(links,key=links.get)
            #print(best_x)
            self.best = translate_parameters(param_key,best_x)
            #print(self.best)
            
            #shutil.copytree(f'{self.workdir}/output/job_{i}', f'{self.workdir}/LF_{self.n}')
            os.mkdir(os.path.join(f"{self.workdir}/LF_{self.n}"))
            np.savetxt(f"{self.workdir}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/emod.best.txt", self.best)
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.best.txt", self.best)
            self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv")
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            
            plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get)],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
            
        elif max(links.values())[0] > self.ymax:
            self.ymax = max(links.values()) #weighted_lf
            best_x = max(links,key=links.get)
            #print(best_x)
            self.best = translate_parameters(param_key,best_x)
            #print(self.best)
                       
            #shutil.copytree(f'{self.workdir}/output/job_{i}', f'{self.workdir}/LF_{self.n}')
            os.mkdir(os.path.join(f"{self.workdir}/LF_{self.n}"))
            np.savetxt(f"{self.workdir}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/emod.best.txt", self.best)
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.txt", self.ymax)
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.best.txt", self.best)
            self.best.to_csv(f"{self.workdir}/LF_{self.n}/emod.best.csv")
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.txt", np.array(self.ymax) * self.objectiveFunction.weights.cpu().numpy())
            #np.savetxt(f"{self.workdir}/LF_{self.n}/emod.ymax.weighted.sum.txt", self.objectiveFunction(torch.tensor(np.array(self.ymax))))
            
            plot_all_comparisons(param_sets_to_plot=[max(pset,key=pset.get)],plt_dir=os.path.join(f"{self.workdir}/LF_{self.n}"))
            self.n += 1
            np.savetxt(f"{self.workdir}/emod.n.txt", [self.n])
        
                      
        
        return torch.tensor(xc,dtype=torch.float64), torch.tensor(yc)

output_dir = "output/Tobias"
best_dir = "checkpoints/Tobias" 

# Delete everything and restart from scratch 
# Comment this line to restart from the last state instead
if os.path.exists(output_dir): shutil.rmtree(output_dir)
if os.path.exists(best_dir): shutil.rmtree(best_dir)

# at beginning of workflow, cleanup all sbatch scripts for analysis
clean_analyzers()

problem = Problem(workdir="output/Tobias")

# Create the GP model
# See emulators/GP.py for a list of GP models
# Or add your own, see: https://botorch.org/docs/models
model = ExactGP(noise_constraint=GreaterThan(1e-6))

# Create and combine multiple batch generators
#batch_size 64 when running in production
tts = TurboThompsonSampling(batch_size=16, n_candidates=1000, failure_tolerance=4, dim=problem.dim) #64
#ei = ExpectedImprovement(batch_size=50, num_restarts=20, raw_samples=1024, dim=problem.dim)
batch_generator = tts#ei#BatchGeneratorArray([tts, ei])

# Create the workflow
bo = BO(problem=problem, model=model, batch_generator=batch_generator, checkpointdir=output_dir, max_evaluations=6)

# Sample and evaluate sets of parameters randomly drawn from the unit cube

#bo.initRandom(2)

bo.initRandom(3, n_batches = 1, Xpriors = [[0.000000000765, 0.2, 0.002, 0.0615,
                                           0.708785, 0.77282, 0.635, 0.5183, 0.5886,
                                           0.15, 0.511735322,
                                           0.005, 0.4151, 0.5, 0]])

# Run the optimization loop
bo.run()

x=pd.read_csv("test_parameter_key.csv")
parameter_labels=x['parameter_label'].to_list()

# Plot
plot_runtimes(bo), 
plt.savefig('output/runtime', bbox_inches="tight")                                                                    
plot_MSE(bo,n_init=1)
plt.savefig('output/mse', bbox_inches="tight")                                                                           
plot_convergence(bo, negate=True, ymin=-20000, ymax=-1000)
plt.savefig('output/convergence', bbox_inches="tight")
plot_prediction_error(bo) 
plt.savefig('output/pred_error', bbox_inches="tight")                                                
plot_X_flat(bo, param_key = x, labels=parameter_labels)
plt.savefig('output/x_flat', bbox_inches="tight")
#plot_space(bo, -5**2, 0, labels="X")
#plt.savefig('output/space', bbox_inches="tight")
#plot_y_vs_posterior_mean(bo,n_init=1)
#plt.savefig('output/posterior_mean', bbox_inches="tight")
