# Create RNN 
"""
Created on Fri Mar 16 18:22:11 2018

@author: Alfredo
"""
#### Importing

import copy
import pyximport; pyximport.install()
from globalVariables import NET_SIZE, CON_MUT_RATE, ACT_FUN_HIDDEN, ACT_FUN_OUTPUT
from constants import TEACHER_FUNCTION, IS_BATCH, TRAINING_SIZE, NUM_ENV_PER_GENERATION, NET_STRUCTURE, INCL_BIASES, MUT_SIZE, PLASTICITY_COST, DISTANCE_METRIC, SEL_STRENGTH, CROSSOVER_RATE, INCL_CROSSOVER, NUM_GENS_PER_ENVIRONMENT, LAMBDA, FORWARD_CYCLES
#from numbers import Number
from matplotlib import pyplot as plt
from numpy import random, NAN, exp, linalg, argmax, mean, max, min, array, zeros
from supFunctions import dist, generate_training, get_norm_cumsum


class Network(object):
    # ATTRIBUTES
    # weights: between layer weights
    # bias: layer biases
    # inclBiases: specifies if the network includes biases
    # actFunHidden: activation function for hidden layers
    # actFunOutput: activation function for output layer
    # numLayers: number of layers in the network
    
    # CONSTRUCTOR
    def __init__(self, structure=NET_STRUCTURE, inclBiases=INCL_BIASES,
                 forwardCycles=FORWARD_CYCLES,
                 actFunHidden=ACT_FUN_HIDDEN, actFunOutput=ACT_FUN_OUTPUT):
        # structure: list of integers, each element indicates the size of its 
        # respective layer ordered from input to output
        # first number determines input size, last number output size, minimum
        # list lenght 2
        # inclBiases: boolean, specifies whether to generate biases as well as
        # links
        # actFunHidden: activation function for hidden layers
        # actFunOutput: activation function for output layer
        # initScalingFactor: numeric, range of initialization values generated 
        # for links and weights
        
        # arguements validation      
        self.numLayers = len(structure)
        #if numLayers < 2:
        #    raise TypeError("Invalid argument: The network requires at leat \
        #                        2 layers. Add more than 1 layer to structure")
        #if not all(isinstance(n, int) for n in structure):
        #    raise TypeError("Invalid argument: Structure needs to be a list \
        #                        of integers")
        #if not isinstance(initScalingFactor, Number):
        #    raise TypeError("Invalid argument: initScalingFactor needs to \
        #                        be a number")
        #if not isinstance(inclBiases, bool):
        #    raise TypeError("Invalid argument: inclBiases needs to be a \
        #                        boolean")
        
        # network construction
        # bias creation
        if inclBiases:
            self.bias = [None] * (self.numLayers - 1)
            for i in range(1, self.numLayers):
                self.bias[i-1] = zeros(shape = structure[i])

        # weight matrices creation
        self.weights = [None] * (self.numLayers - 1)
        for i in range(self.numLayers - 1):
            self.weights[i] = zeros(shape = (structure[i], structure[i+1]))
        # Create random weights for testing
        for i in range(self.numLayers - 1):
            self.weights[i] = random.uniform(
                    high=2, low=-2,
                    size= (structure[i], structure[i+1]))

        
        # store activation functions
        self.actFunHidden = actFunHidden
        self.actFunOutput = actFunOutput
        
        # store number of development iterations
        self.forwardCycles = forwardCycles
        
        # store support variables
        self.inclBiases = inclBiases 
        
    # METHODS   
    def forward(self, inputValues):
        # calculates the output vector from given input vector inputValues 
        # and returns the outputValues
        # inputValues: input vector, must be of type np.array and of same 
        # length as input layer
        
        outputValues = inputValues
        # calculate the activity of the hidden layers 
        for j in range(self.forwardCycles):
            for i in range(self.numLayers - 2):
                outputValues = outputValues.dot(self.weights[i]) + (self.bias[i] if self.inclBiases == True else 0)
                outputValues = self.actFunHidden(outputValues)
        # calculate the active of the output layer
        outputValues = outputValues.dot(self.weights[-1]) + (self.bias[-1] if self.inclBiases == True else 0)
        outputValues = self.actFunOutput(outputValues)
        return outputValues