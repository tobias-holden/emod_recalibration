from run_sims import submit_sim
from run_analyzers import run_analyzers
from get_eradication import get_eradication
from compare_to_data.run_full_comparison import compute_LL_across_all_sites_and_metrics
import argparse
import params as params
import manifest as manifest
import os, sys, shutil
import time
import matplotlib.pyplot as plt
from load_inputs import load_sites
import pandas as pd
import numpy as np

from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval, GreaterThan, LessThan

from translate_parameters import translate_parameters, get_initial_samples

sites = load_sites()

def my_func(X):
  # Supply parameters to X
  get_eradication(manifest.use_local_eradication)
  param_key=pd.read_csv("test_parameter_key.csv")
  df = pd.DataFrame({'parameter':[], 'unit_value': [], 'emod_value':[], 'type':[], 'param_set':[]})
  i=1
  for x in X:
      a = translate_parameters(param_key,x)
      a['param_set'] = np.repeat(i,len(a))
      i=i+1
      df = pd.concat([df,a])
 

  for my_site in sites:
      if os.path.exists(os.path.join(manifest.simulation_output_filepath,my_site)):
          shutil.rmtree(os.path.join(manifest.simulation_output_filepath,my_site))
      submit_sim(site=my_site, X=df)
    
  #for my_site in sites:
   #   run_analyzers(site=my_site)

  while True:
      outputs = []
      for my_site in sites:
          outputs.append(os.path.exists(os.path.join(manifest.simulation_output_filepath,my_site,'finished.txt')))
      if all(outputs):#os.path.exists(manifest.simulation_output_filepath,my_site): 
          Y = compute_LL_across_all_sites_and_metrics(numOf_param_sets=len(X))
          break
      time.sleep(120)   
   
  print(Y)
    
  return(Y)

if __name__ == '__main__':
    param_key=pd.read_csv("test_parameter_key.csv")
    X = get_initial_samples(param_key, 100)
    X = X[1:5]
    my_func(X)
