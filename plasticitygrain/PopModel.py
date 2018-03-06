# -*- coding: utf-8 -*-
################################################################################
######################### NAMING CONVENTIONS ###################################
################################################################################

# Constants are in capital letters separated by underscores
# Classes are annotated using CamelCase notation
# Functions are annotated in lowercase separted by underscores
# Variables are annotated in pascalNotation

################################################################################
############################ IMPORTING #########################################
################################################################################

import os
import cProfile as profile
from numpy.random import seed
from classes import Environment
from experiments import run_instance


################################################################################
################################ MAIN ##########################################
################################################################################

#clear screen
os.system('cls')

#run

def main(): 
    seed(1)
    env = Environment() 
    path = '/Plasticity/Experiment_Pop_Online_Test'
    run_instance(path, env)
    
if __name__ == '__main__':
    #main()
    profile.run('main()')


        

