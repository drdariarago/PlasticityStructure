################################################################################
############################# IMPORTING ########################################
################################################################################

import os
import shelve
import numpy as np
import constants as const
import pickle
from classes import Population
from supFunctions import plot_performance

################################################################################
############################## EXPERIMENTS #####################################
################################################################################

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def save_global_vars_txt(path):
    tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]
    with open(path + 'global_vars.txt', 'w') as output: 
        for key in tmpDir:
            try:
                output.write(key + ': ' + str(getattr(const,key)) + '\n')  
            except TypeError:
                print('ERROR writing')                

def save_global_vars(path):
    # save global variables
    
    #using pickles
    #tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]

    #with open(filename, 'wb') as output:     # 'wb' instead 'w' for binary file
    #    for key in tmpDir:
    #        print key
    #        pickle.dump(getattr(const,key), output, pickle.HIGHEST_PROTOCOL)

    #using shelve
    tmpShelf = shelve.open(path + 'global_vars.out','n') # 'n' for new    
    tmpDir = [item for item in dir(const) if not item.startswith("__") and item.isupper()]
    for key in tmpDir:
        try:
            tmpShelf[key] = getattr(const, key)
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    tmpShelf.close()                  

def load_global_vars(filename):
    # load global variables
    
    #using shelve
    tmpShelf = shelve.open(filename) 
    for key in tmpShelf:
        try:
            setattr(const,key,tmpShelf[key])
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    tmpShelf.close()  

def run_instance(fileDir, env):
    #run a simulation instance

    #initialisation                                                                                                
    POP = Population(const.POP_SIZE)
    performance = np.empty(shape = (int(const.GENERATIONS/env.numGensPerEnvironment),4), dtype='float64') 
    
    #main
    for gener in range(const.GENERATIONS):
    
        # print progress (%)
        if (float(gener)/const.GENERATIONS*100) % 2 == 0:
            print("%.2f"% (float(gener)/const.GENERATIONS*100) + "%")    
    
        # change environment
        env.generate_environment(gener)
    
        # next generation        
        POP.next_gen(envCues=env.get_cues(), target=env.get_target())    
    
        # store performance evaluation
        if gener % env.numGensPerEnvironment == 0:
            performance[int(gener / env.numGensPerEnvironment),:] = np.array([POP.get_top_fitness(), POP.get_mean_fitness(), 
            POP.get_top_training_performance(env), POP.get_mean_training_performance(env)])
    
    #save data
    #generate directory
    path = os.path.abspath(fileDir) + '/'
    print(path)
    ensure_dir(path)
    
    #store simulation specifications - global variables
    save_global_vars(path)
    
    #store simulation specifications - global variables - txt file
    save_global_vars_txt(path)
    
    #store population
    #save_object(POP, path + 'population.p')
        
    #store and show figures    
    plot_performance(performance,filePath=path,saveData = True, showFig = False, saveFig = True)
    POP.get_best_individual().plot_reaction_norm(env, filePath=path, showFig = False, saveFig = True)   
    ind = POP.get_best_individual().get_weights()
    print(ind)
    
    