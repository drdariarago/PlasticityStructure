################################################################################
############################# IMPORTING ########################################
################################################################################


import pickle
from matplotlib import pyplot as plt
from numpy import exp, asarray, tanh, dot, random, sum, abs, float, cumsum

################################################################################
######################### SUPPORT FUNCTIONS ####################################
################################################################################
    
def identityFunction(x):
    # identity function
    return x

def logistic(x):
    # logistic function for np arrays
    return 1/(1+exp(-x))

def calc_mut_rate(netSize, indMutRate):
    # returns the rate of mutation per link given network size and the mutation
    # rate for each individual
    return (1.0/netSize) * indMutRate
        
def calc_net_size(netStructure, inclBiases):
    # returns the total number of connections for a given a network structure
    # netStructure: vector with one element per network layer from input to 
    # output, must contain integers. Each number defines the size of the layer, 
    # can be list of array, minimum length 2.
    # inclBiases: boolean, determines if the networks have bias terms as well as 
    # links
    netStructure = asarray(netStructure)
    return sum(netStructure[:-1] * netStructure[1:]) + ( sum(netStructure[1:]) if inclBiases else 0 )
    
    
def calc_net_probabilities(netStructure, inclBiases):
    pWeights = [a*b for a,b in zip(netStructure[:-1],netStructure[1:])]
    if inclBiases:
        pWeights.extend(netStructure[1:])
    return [x/float(sum(pWeights)) for x in pWeights]
    
def get_norm_cumsum(inputArray):
    #inputArray is 1D array
    return cumsum(inputArray/float(sum(inputArray)))    
    
def get_act_func(name):
    #returns the activation function of the respective name
    if name == 'tanh':
        func = tanh
    elif name == 'identity':
        func = identityFunction
    elif name == 'logistic':
        func = logistic
    return func
    
def dist(x, y, distType):
    # returns the distance between x and y using distType measure
    if distType == 'euclidean':
        d = sum(abs(x - y) ** 2, axis=1) ** .5
    elif distType == 'dotProd':
        d = - dot(x.T, y)
    return d
    
def generate_training(trainSize, teacherFunction):
    # generate training set of size trainSize using the teacherFunction
    # returns two matrices (np arrays) for the inputs and the outputs 
    # respectively
    # each row corresponds to a training sample.
    if teacherFunction == 'linear':
        # generate trainSize points drawn uniformly from [a,b]
        #x1,x2 = 0,2
        #inputArray = random.uniform(low=x1, high=x2, size=(trainSize,1)) 

        # generate trainSize points drawn normal distribution N(1,sigma)
        sigma = 0.4
        inputArray = random.normal(1,sigma, size = (trainSize,1))

        # apply linear transformation of slope a and intercept b
        a,b = -2,4
        outputArray = a * inputArray + b
    return inputArray, outputArray
    	
def plot_performance(performData, showFig = False, saveFig = False, saveData = False, filePath = ''):
    
    fig = plt.figure(figsize=(10,10))
    
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['Top Fitness', 'Mean Fitness', 'Top Performance', 'Mean Performance']
    
    for i in range(performData.shape[1]):
        plt.plot(performData[:,i], color = colors[i], label = labels[i])
      
    plt.legend(loc = 'best')
    plt.xlabel('Generations')
    plt.ylabel('Performance')  
    plt.title('Population performance over evolutionary time')
    plt.ylim([0, 1.1])
    
    #show figure
    if showFig:
        plt.show()
        
    #save figure
    if saveFig:
        fig.savefig(filePath + 'performance.png')
        
    #save data
    if saveData:
        with open(filePath + 'performance.p', 'wb') as output:
            pickle.dump(performData, output, pickle.HIGHEST_PROTOCOL)
        