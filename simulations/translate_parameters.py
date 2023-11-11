import os
import sys
import numpy as np
import pandas as pd

#### Read in Parameter Key
key_path = 'test_parameter_key.csv'
parameter_key = pd.read_csv(key_path)

#### Define Parameter Translator

def translate_parameters(key, guesses):
    
    result = "Done" 
    output = pd.DataFrame({"parameter":[], 
                           "param_set": [],
                           "min":[], "max":[],"transformation":[], "type":[], "team_default":[],
                           "unit_value":[],
                           "emod_value":[]}, columns=['parameter','param_set', 'unit_value', 'emod_value',"min","max","team_default","transformation","type"])
    if(len(guesses) % len(key.index) != 0):
        result="Error: Number of guesses must equal the number of parameters in the key"
        print(result,"(",len(key.index),")")
        return
    #if(any(guesses<0) or any(guesses>1)):
     #   result="Error: Guesses must lie between 0 and 1"
      #  print(result)
       # return
    
    for index, row in key.iterrows():
        
        # Scale to parameter range
        value = guesses[index]*row['max']
        
        # Apply Transformations
        if(row['transform']=='log'):
            value = row['max'] * guesses[index]**2  # fixme, we will need to update this to ensure it is doing the correct log transformation
        if(row['transform']=='IIVT'):
            if(guesses[index] <= float(1/3)):#float(0.5)):#
                value = "NONE"
            elif(guesses[index]<= float(2/3)):
                value = "PYROGENIC_THRESHOLD_VS_AGE"
            else: 
                value = "PYROGENIC_THRESHOLD_VS_AGE_INC"
        
        # Restrict Min/Max
        if(row['transform'] != 'IIVT'):
            if( value < row['min']):
                value = row['min']
            if(value > row['max']):
                value = row['max']
        
        # Convert Data Types
        if(row['type'] == 'integer'):
            value = np.trunc(value)
        
        default = row['team_default']
        if (default==''):
            default = row['emod_default']
            
        
        
        new_row = pd.DataFrame({"parameter": [row['parameter_name']], 
                                #"param_set": [row['param_set']],
                                "team_default":[default],
                                "unit_value": [guesses[index]],
                                "min": [row['min']],
                                "max": [row['max']],
                                "transformation": [row['transform']],
                                "type":[row['type']],
                                "emod_value": [value]}, columns=['parameter','unit_value', 'emod_value',"min","max","team_default","transformation","type"])
        
        output = pd.concat([output, new_row])
        
        #print(row['parameter_name'])
        #print(guesses[index],'-->',value)
        
    output = output.reset_index(drop=True)
    
    #print(result)
    #print(output[['parameter','unit_value','emod_value','type']])    
    return(output)

#### Generate Initial Samples

def get_initial_samples(key, size=1):
    n_param = len(key.index)
    values = np.linspace(0,1,size)
    np.random.shuffle(values)
    values = list(values)
    for i in range(n_param-1):
        x=np.linspace(0,1,size)
        np.random.shuffle(x)
        x=list(x)
        values.extend(x)

    values = np.array(values).reshape(n_param,size).transpose()
    
    #values['param_set'] = np.trunc(values.index/15)+1
    #print(values.shape)
    print(values)
    return(values)
    


if __name__ == '__main__':
    size = 10
    #initial_samples = get_initial_samples(parameter_key, size)
    #print(initial_samples)
    param_key=pd.read_csv("test_parameter_key.csv")
    #param_guess=pd.read_csv("param_guess.csv")
    #translate_parameters(param_key,param_guess)
    best=np.loadtxt("checkpoints/emod/LF_6/emod.best.txt").astype(float)
    print(best)
    translated_best = translate_parameters(param_key,best)
    print(translated_best)
