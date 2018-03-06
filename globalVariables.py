################################################################################
############################# IMPORTING ########################################
################################################################################

from constants import NET_STRUCTURE, INCL_BIASES, IND_MUT_RATE, ACT_FUN_OUTPUT, ACT_FUN_HIDDEN
from supFunctions import calc_net_probabilities, calc_net_size, calc_mut_rate, get_act_func

################################################################################
######################### GLOBAL VARIABLES #####################################
################################################################################

NET_SIZE = calc_net_size(NET_STRUCTURE, INCL_BIASES)
CON_MUT_RATE = calc_mut_rate(NET_SIZE, IND_MUT_RATE)
NET_MUT_PROBABILITIES = calc_net_probabilities(NET_STRUCTURE,INCL_BIASES) 
ACT_FUN_OUTPUT = get_act_func(ACT_FUN_OUTPUT)
ACT_FUN_HIDDEN = get_act_func(ACT_FUN_HIDDEN)
