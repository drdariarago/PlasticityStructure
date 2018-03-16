################################################################################
############################# CONSTANTS ########################################
################################################################################

# NET_STRUCTURE: vector with one element per network layer from input to output,
# must contain integers. Each number defines the size of the layer, can be list
# of array, minimum length 2.
NET_STRUCTURE = [2, 2]

# INCL_BIASES: boolean, determines if the networks have bias terms
INCL_BIASES = True

# ACT_FUN_HIDDEN: activation function for hidden layers
ACT_FUN_HIDDEN = 'tanh'

# ACT_FUN_OUTPUT: activation function for output layer
ACT_FUN_OUTPUT = 'step'

# DISTANCE_METRIC: function used to calculate distance between phenotype and
# target
DISTANCE_METRIC= 'euclidean'

# SEL_STRENGTH: selection strength. The lower the value, the higher the strength of selection is.
SEL_STRENGTH = 0.1 #0.1

# IND_MUT_RATE: mutation rate per capita
IND_MUT_RATE = .1

# MUT_SIZE: change in weight per mutation (std: normal distribution)
MUT_SIZE = .02

# INCL_CROSSOVER: boolean, determines whether crossover is taken into account
INCL_CROSSOVER = False

# CROSSOVER_RATE: proportion of weights and biases inhertied from partner
CROSSOVER_RATE = 0.5

# PLASTICITY_COST: cost of plasticity
# penalty for individuals with plastic responses, applied to fitness calculations
PLASTICITY_COST = 0.01

# TRAINING_SIZE: number of samples in the training set
TRAINING_SIZE = 10

# POP_SIZE: number of individuals in population
POP_SIZE = 100

# NUM_GENS_PER_ENVIRONMENT: number of generations per environmental changes
NUM_GENS_PER_ENVIRONMENT = 2000

# NUM_ENV_PER_GENERATION: number of environments per generation
NUM_ENV_PER_GENERATION = 1

# GENERATIONS: number of generations
GENERATIONS = 2000

# IS_BATCH: boolean, determines if fitting with batch (True) or not (False)
# if True, no need to update training set every time
IS_BATCH = False

# TEACHER_FUNCTION: the function that describes the underlying problem structure
TEACHER_FUNCTION = 'linear'

#ELITIST_PROPORTION: the proportion of the population retained by the elitist selection
ELITIST_PROPORTION = 0.1

#LAMBDA: Average number of mutations per individual
LAMBDA = 0.2

# FORWARD_CYCLES: Number of cycles of RNN development
FORWARD_CYCLES = 20