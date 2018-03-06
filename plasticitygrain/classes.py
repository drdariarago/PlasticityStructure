################################################################################
############################# IMPORTING ########################################
################################################################################

import copy
import pyximport; pyximport.install()
from globalVariables import NET_SIZE, CON_MUT_RATE, ACT_FUN_HIDDEN, ACT_FUN_OUTPUT
from constants import TEACHER_FUNCTION, IS_BATCH, TRAINING_SIZE, NUM_ENV_PER_GENERATION, NET_STRUCTURE, INCL_BIASES, MUT_SIZE, PLASTICITY_COST, DISTANCE_METRIC, SEL_STRENGTH, CROSSOVER_RATE, INCL_CROSSOVER, NUM_GENS_PER_ENVIRONMENT, LAMBDA
#from numbers import Number
from matplotlib import pyplot as plt
from numpy import random, NAN, exp, linalg, argmax, mean, max, min, array, zeros
from supFunctions import dist, generate_training, get_norm_cumsum

################################################################################
############################## CLASSES #########################################
################################################################################

############################## NETWORK #########################################

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
        
        # store activation functions
        self.actFunHidden = actFunHidden
        self.actFunOutput = actFunOutput
        
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
        for i in range(self.numLayers - 2):
            outputValues = outputValues.dot(self.weights[i]) + (self.bias[i] if self.inclBiases == True else 0)
            outputValues = self.actFunHidden(outputValues)
        # calculate the active of the output layer
        outputValues = outputValues.dot(self.weights[-1]) + (self.bias[-1] if self.inclBiases == True else 0)
        outputValues = self.actFunOutput(outputValues)
        return outputValues

    #COPY FUNCTIONS
    #def copy(self, ind):
    #    self.weights = [nWeights.copy() for nWeights in ind.weights]
    #    self.bias = [nBias.copy() for nBias in ind.bias]
        
    def copy(self):
        newNet = Network.__new__(Network)
        newNet.__class__ = self.__class__
        newNet.weights = [nWeights.copy() for nWeights in self.weights]
        newNet.bias = [nBias.copy() for nBias in self.bias]
        newNet.inclBiases = self.inclBiases
        newNet.numLayers = self.numLayers
        newNet.actFunHidden = self.actFunHidden
        newNet.actFunOutput = self.actFunOutput
        return newNet        

############################# INDIVIDUAL #######################################

class Individual(object):
    # ATTRIBUTES
    # phen: Adult phenotype (protected)
    # fitness: Fitness (protected)
    # network: Object of class Network (protected)
    # conMutRate: Mutation rate per weight
    # netSize
    
    # CONSTRUCTOR
    def __init__(self):
        self.__fitness = 1
        self.__phen = NAN
        self.__network = Network()
        self.__conMutRate = CON_MUT_RATE
        self.__netSize = NET_SIZE
    
    # METHODS    
    def develop(self, envCues):
        # develop: Activation function, generates attribute __phen 
        # (phenotype) based on environmental input (envCues)
        # use function get_phenotype to return calculated phenotypes
        # envCues: environmental cues, np.array vector of size equal to 
        # input layer
        self.__phen = self.__network.forward(envCues)

    def mutate(self):
        
        # mutate: mutation function     
        for weights in self.__network.weights:
            mask = random.rand(*weights.shape) < self.__conMutRate
            if any(mask):
                if len(weights) == 1:
                    weights += random.normal(0, MUT_SIZE)
                else:    
                    weights[mask] += random.normal(loc=0, 
                    scale=MUT_SIZE, size=weights[mask].shape)
        if self.__network.inclBiases:
            for bias in self.__network.bias:
                mask = random.rand(*bias.shape) < self.__conMutRate
                if any(mask):
                    if len(bias) == 1:
                        bias += random.normal(0, MUT_SIZE)
                    else:
                        bias[mask] += random.normal(loc=0, 
                        scale=MUT_SIZE, size=bias[mask].shape)
 
         
    def eval_performance(self, phen, target, plasticityCost = PLASTICITY_COST):
        # evaluate performance of the organism compared to a set of target phenotypes
        # target: selective environment, vector of size equal to phenotype
        benefit = - dist(phen, target, DISTANCE_METRIC).mean() 
        cost = linalg.norm(self.__network.weights[0].flatten(), ord = 2)
        performance = benefit - plasticityCost * cost
        return exp(performance / (2 * SEL_STRENGTH))
        
    def eval_fitness(self, target):
        # evaluate fitness of the organism
        # target: selective environment, vector of size equal to phenotype
        self.__fitness = self.eval_performance(phen = self.__phen, target = target)

    def eval_training_performance(self, env):
        # evaluate performance of the organism across the entire environmental range of the experiment 
        tmpPhen = self.__network.forward(inputValues = env.get_training_input())
        return self.eval_performance(phen = tmpPhen, target = env.get_training_output(), plasticityCost = 0)
        
    def crossover(self, partner):
        # perform crossover between parents. 
        # crossover is applied on each column, including the corresponding bias
        # node when appropriate.
        for i in range(len(self.__network.weights)):
            mask = random.rand(self.__network.weights[i].shape[1]) > CROSSOVER_RATE
            self.__network.weights[i][:,mask] = partner.__network.weights[i][:,mask]
            if self.__network.inclBiases:
                self.__network.bias[i][mask] = partner.__network.bias[i][mask]

    # GET FUNCTIONS
    def get_phenotype(self):
        # return phenotype
        return self.__phen

    def get_fitness(self):
        # return fitness
        return self.__fitness
        
    def get_network(self):
        # return network
        return self.__network
        
    def get_weights(self):
        ### return weights
        return self.__network.weights
        
    def get_bias(self):
        if INCL_BIASES == True:
            return self.__network.bias
        else:
            return 0
        
    #COPY FUNCTIONS
    def copy(self):
        newInd = Individual.__new__(Individual)
        newInd.__class__ = self.__class__
        newInd.__network = self.__network.copy()
        newInd.__phen = self.__phen
        newInd.__conMutRate = self.__conMutRate
        newInd.__fitness = self.__fitness
        newInd.__netSize = self.__netSize
        return newInd
            
    # PLOT FUNCTIONS
    def plot_reaction_norm(self, env, showFig = False, saveFig = False, filePath = ''):
        # plot reaction norm
        self.develop(env.get_training_input())
        
        fig = plt.figure(figsize=(10,10))

        plt.scatter(env.get_training_input(), self.get_phenotype(), label = 'Developed phenotype')
        plt.scatter(env.get_training_input(), env.get_training_output(), marker='x', color='red', label = 'Target phenotype')
        
        plt.xlabel('Environmental cue')
        plt.ylabel('Phenotype')
        plt.title('Reaction Norm')
        plt.legend(loc = 'best')
        
        #show figure
        if showFig:
            plt.show()
        
        #save figure
        if saveFig:
            fig.savefig(filePath + 'reaction_norm.png')
        
        
############################# POPULATION #######################################

class Population(object):
    # ATTRIBUTES
    # ind: List of individuals (protected)
    # popSize: number of individuals
    
    # CONSTRUCTOR
    def __init__(self, popSize = 1): #default population size is 1. Hill climbing approximation.
        self.__popSize = popSize
        self.__ind = [Individual()] * self.__popSize
    
    # METHODS
    def next_gen(self, envCues, target):
        # iterate development/selection/reproduction for one generation
        # develop and evaluate fitness of individuals in current population
        if len(self.__ind) > 1:
        
            # select best individuals and create new population
            #self.__ind = [self.create_child() for _ in range(self.__popSize)]
            self.__ind = self.select_new_pop()
            
            #mutate
            self.mutate()
            
            #develop
            self.develop(envCues)
            
            #eval fitness
            self.eval_fitness(target)
            
            #update fitness
            self.__cumFit = get_norm_cumsum(self.get_fitness())
        
        else:
        # Iterate development and selection
            self.__ind[0].develop(envCues)
            self.__ind[0].eval_fitness(target)
        # Copy individual to child, mutate develop and eval fitness
            child = copy.deepcopy(self.__ind[0])
            child.mutate()
            child.develop(envCues)
            child.eval_fitness(target)
        # If the child is fitter than the parent change population
            if child.get_fitness() > self.__ind[0].get_fitness():
                self.__ind[0] = copy.deepcopy(child)
 
    def mutate(self):
        # mutate each individual in the population
        for member in self.__ind:
            member.mutate()           
        
    def develop(self, envCues):
        # develop all individuals in the population
        for member in self.__ind:
            member.develop(envCues)

    def eval_fitness(self, target):
        # evaluate the fitness of each individual in the population
        for member in self.__ind:
            member.eval_fitness(target)
			
    def create_child(self):
        # create individuals for new population
        if INCL_CROSSOVER:
            parent1, parent2 = self.get_parents()
            child = copy.deepcopy(parent1)            
            child.crossover(parent2)
        else:
            child = Individual()
            child.copy(self.select_individual())
        return child
    
    def select_individual(self):
        # select individuals using fitness proportion selection   
        pick = random.random()
        for i in range(self.__popSize):
            if pick <= self.__cumFit[i]:
                return self.__ind[i]

    def select_new_pop(self):
        # select individuals using fitness proportion selection
        p = self.get_fitness()
        picks = random.choice(range(self.__popSize), size = self.__popSize, p = p/float(sum(p))) 
        return [self.__ind[k].copy() for k in picks]        
        
    def get_parents(self):
        # return parents based on proportional fitness, returns individual type 
        # objects
        parent1 = self.select_individual()      
        while True:
            parent2 = self.select_individual()
            if not parent1 == parent2:
                break
        return self.__ind[parent1], self.__ind[parent2]
        
    # SAVE
    def save(self):
        # Save population
        return self          
            
    ### GET FUNCTIONS
    def get_individual(self):
        # return the list of individuals
        return self.__ind

    def get_fitness(self):
        # return a list of individuals' fitness
        return array([ind.get_fitness() for ind in self.__ind])
        
    def get_phenotype(self):
        # return list of phenotypes
        return [self.__ind[i].get_phenotype() for i in range(self.__popSize)]

    def get_best_individual(self):
        # return the index / position of the best individual in the population
        return self.__ind[argmax(self.get_fitness())]

    def get_top_fitness(self):
		# evaluate the best fitness among all individuals
        return max(self.get_fitness())
			
    def get_mean_fitness(self):
        # evaluate mean fitness of all individuals
        return mean(self.get_fitness())
		
    def get_train_performance(self, env):
        # return a list of individuals' performance across the whole training set
        return [self.__ind[i].eval_training_performance(env) for i in range(self.__popSize)]

    def get_top_training_performance(self, env):
		# evaluate the best performance among each individual across the whole training set
        return max(self.get_train_performance(env))
		
    def get_mean_training_performance(self, env):
		# evaluate mean performance among each individual across the whole training set
        return mean(self.get_train_performance(env))	
	

class Environment(object):
    
    # CONSTRUCTOR
    def __init__(self):
        # numGensPerEnvironment: number of generations per environment
	# numEnvPerGeneration: number of environment per each generation
        self.__trainingInput, self.__trainingOutput = generate_training(trainSize=TRAINING_SIZE, teacherFunction=TEACHER_FUNCTION)    
        self.__cues, self.__target = self.__trainingInput, self.__trainingOutput
        self.numGensPerEnvironment = NUM_GENS_PER_ENVIRONMENT
        self.numEnvPerGeneration = NUM_ENV_PER_GENERATION
        self.trainingSize = TRAINING_SIZE
        self.isBatch = IS_BATCH       
    
    # METHODS
    def generate_environment(self, gener):
        # gener: number of current generation
        
        if not self.isBatch and gener % self.numGensPerEnvironment == 0:
            choices = random.choice(self.trainingSize, self.numEnvPerGeneration, replace=False)
            self.__cues, self.__target = self.__trainingInput[choices,], self.__trainingOutput[choices,]
            
    ### GET FUNCTIONS
    def get_training_input(self):
        # return the training input
        return self.__trainingInput

    def get_training_output(self):
        # return the training output
        return self.__trainingOutput
                
    def get_cues(self):
        # return a list of inputs
        return self.__cues

    def get_target(self):
        # return desired outputs
        return self.__target        
            