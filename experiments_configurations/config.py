# TODO Create a dic here to place all the parameters for each task
from abc import abstractmethod
from enum import Enum
import numpy as np

class EnvironmentNames(Enum):
    COFFEE = "our_gym_environments/CoffeeTaskEnv-v0"
    TAXI_SMALL = "our_gym_environments/TaxiSmallEnv-v0"
    TAXI_BIG = "our_gym_environments/TaxiBigEnv-v0"

class EnvironmentType(Enum):
    STOCHASTIC = "stochastic"
    DETERMINISTIC = "deterministic"

class ActionCountStrategy(Enum):
    Relational = "relational"
    Original = "original"

class EpisodeStateInitialization(Enum):
    RELATIONAL_EPISODE_NUMBER = "relational episode number"
    SAME = 0 #Use 488 for Taxi task
    RANDOM = 1
    EPISODE_NUMBER = "episode number"

class DecayRate(Enum):
    FAST = 0.01
    NORMAL = 0.025
    SLOW = 0.004

class EpsilonStrategy(Enum):
    FIXED = 'Fixed',  # notice the trailing comma
    DECAYED = 'Decayed'

class EvaluationMetric(Enum):
    EPISODE_REWARD = 'Episode Reward'
    CURRENT_Q = 'Current Q'

class ModelUseStrategy(Enum):
    IMMEDIATE_POSITIVE = 'Immediate positive'
    POSITIVE_OR_NOT_NEGATIVE = 'Positive or not negative'
    DATA_AUGMENTATION = 'Data augmentation'

class ActionSelectionStrategy(Enum):
    EPSILON_GREEDY = 'Epsilon-Greedy'
    MODEL_BASED_EPSILON_GREEDY = 'Model Based Epsilon-Greedy'
    NEW_IDEA = 'New Idea'
    RANDOM_MODEL_BASED = 'Random model based'

class ModelDiscoveryStrategy(Enum):
    NEW_IDEA = 'New Idea'
    EPSILON_GREEDY = 'Epsilon Greddy'
    RANDOM = 'Random'
    LESS_SELECTED_ACTION_EPSILON_GREEDY = 'Less selected action epsilon greedy'
    LESS_VISITED_NEXT_STATE = 'Less visited next state'
    GO = 0
    GU = 1
    BC = 2
    DC = 3
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICK = 4
    DROP = 5

class Times(Enum):
    FOREVER = 'Forever',  # notice the trailing comma
    ONE = '1',

class Step(Enum):
    RL = 'RL'  # notice the trailing comma
    CD = 'CD'
    RL_USING_CD = 'RL using CD'
    RL_FOR_CD = 'RL for CD'
    MODEL_INIT = 'Model Initialization'

class Stage:
    def __init__(self, steps, times):
        self.steps = steps
        self.times = times

class CombinationStrategy:
    def __init__(self, name, stages):
        self.name = name
        self.stages = stages

class AlgConf:
    def __init__(self, alg_name, combination_strategy):
        self.alg_name = alg_name
        self.combination_strategy = combination_strategy

class RLConf(AlgConf):
    def __init__(self, alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization):
        super().__init__(alg_name, combination_strategy)
        self.n = n
        self.alpha = alpha  # learning rate (Use 1 as optimal for deterministic environments, use 0.1 for stochastic)
        self.gamma = gamma  # discount rate
        self.epsilon_start = epsilon_start  # Exploration factor (1 is full exploration, 0 is none exploration)
        self.epsilon_end = epsilon_end  # Exploration factor (1 is full exploration, 0 is none exploration)
        # self.epsilon_end = epsilon_strategy
        # self.decay_rate = decay_rate
        self.rl_action_selection_strategy = rl_action_selection_strategy
        self.episode_state_initialization = episode_state_initialization

class CRLConf(RLConf):
    def __init__(self, alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization, T, th, model_use_strategy, model_discovery_strategy, min_frequency, crl_action_selection_strategy, use_crl_data, model_init_path = None):
        super().__init__(alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization)
        self.T = T
        self.th = th
        self.model_use_strategy = model_use_strategy
        self.model_discovery_strategy = model_discovery_strategy
        self.min_frequency = min_frequency
        self.crl_action_selection_strategy = crl_action_selection_strategy
        self.use_crl_data = use_crl_data
        self.model_init_path = model_init_path

class ExpConf:

    def __init__(self, exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, action_count_strategy, shared_initial_states, alg_confs):
        self.exp_name = exp_name
        self.env_name = env_name
        self.env_type = env_type
        self.trials = trials
        self.smooth = smooth
        self.evaluation_metric = evaluation_metric
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.action_count_strategy = action_count_strategy
        self.shared_initial_states = shared_initial_states
        self.alg_confs = alg_confs

# Instantiations

# Stages
single_stage_1 = Stage([Step.RL], Times.ONE)
single_stage_2 = Stage([Step.MODEL_INIT], Times.ONE)
single_stage_3 = Stage([Step.RL_FOR_CD], Times.ONE)
single_stage_4 = Stage([Step.CD], Times.ONE)

cycle_stage_0 = Stage([Step.RL, Step.CD, Step.RL_USING_CD, Step.CD], Times.FOREVER)
cycle_stage_1 = Stage([Step.CD, Step.RL_USING_CD, Step.RL], Times.FOREVER)
cycle_stage_2 = Stage([Step.CD, Step.RL_USING_CD, Step.CD, Step.RL], Times.FOREVER)
cycle_stage_3 = Stage([Step.RL_USING_CD, Step.CD], Times.FOREVER)

cycle_stage_4 = Stage([Step.RL_USING_CD], Times.FOREVER)
cycle_stage_5 = Stage([Step.RL_FOR_CD, Step.CD, Step.RL_USING_CD, Step.CD], Times.FOREVER)
cycle_stage_6 = Stage([Step.RL, Step.CD], Times.FOREVER)
cycle_stage_7 = Stage([Step.RL_FOR_CD, Step.CD], Times.FOREVER)
cycle_stage_8 = Stage([Step.RL], Times.FOREVER)

cycle_stage_9 = Stage([Step.CD, Step.RL_USING_CD], Times.FOREVER)

# Combination Strategies
combination_strategy_0 = CombinationStrategy("RL, CD, RLusingCD, repeat", [cycle_stage_0])
combination_strategy_1 = CombinationStrategy("RLforCD, CD, RLusingCD, CD, repeat", [cycle_stage_5])
combination_strategy_2 = CombinationStrategy("PGM strategy", [single_stage_1, cycle_stage_2])
combination_strategy_3 = CombinationStrategy("CombStrategy3", [single_stage_1, cycle_stage_5])
combination_strategy_4 = CombinationStrategy("Transfer", [single_stage_2, cycle_stage_4])
combination_strategy_5 = CombinationStrategy("Rl, CD forever", [cycle_stage_6])
combination_strategy_6 = CombinationStrategy("Rl for CD, CD, forever", [cycle_stage_7])
combination_strategy_7 = CombinationStrategy("RL forever", [cycle_stage_8])
combination_strategy_8 = CombinationStrategy("RLforCD and then CD and RL using CD forever", [single_stage_3, cycle_stage_9])

#############################################################
#                  RL Configurations                        #
#############################################################

# RL Configurations (alg_name, combination_strategy, n, alpha, gamma, .epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization):

# DETERMINISTIC ENVIRONMENT WITH DECAYED EPSILON, alpha = 1.0 (optimal)
rl_conf_1 = RLConf("Q-Learning e0.3dec", combination_strategy_7, 0, 1, 0.95, 0.3, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_2 = RLConf("Q-Learning e0.7dec", combination_strategy_7, 0, 1, 0.95, 0.7, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_3 = RLConf("Q-Learning e1.0dec", combination_strategy_7, 0, 1, 0.95, 1.0, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_4 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)

# STOCHASTIC ENVIRONMENT WITH DECAYED EPSILON, alpha = 0.1
rl_conf_5 = RLConf("Q-Learning e0.3dec", combination_strategy_7, 0, 0.1, 0.95, 0.3, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_6 = RLConf("Q-Learning e0.7dec", combination_strategy_7, 0, 0.1, 0.95, 0.7, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_7 = RLConf("Q-Learning e1.0dec", combination_strategy_7, 0, 0.1, 0.95, 1.0, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_8 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 0.1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)


# DETERMINISTIC ENVIRONMENT WITH FIXED EPSILON
rl_conf_9 = RLConf("Dyna Q n5", combination_strategy_7, 5, 1, 0.95, 0.1, 0.1,  ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_10 = RLConf("Dyna Q n10", combination_strategy_7, 10, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_11 = RLConf("Dyna Q n20", combination_strategy_7, 20, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_12 = RLConf("Dyna Q n50", combination_strategy_7, 50, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)

RL_DETERMINISTIC_CONFS = [rl_conf_1, rl_conf_2, rl_conf_3, rl_conf_4]
RL_STOCHASTIC_CONFS = [rl_conf_5, rl_conf_6, rl_conf_7, rl_conf_8]


#############################################################
#                          COFFEE TASK                      #
#############################################################

#############################################################
# ZERO experiment to find the Ground Truth Models           #
#############################################################

TRIALS = 0
T = 60
exp_coffee_0 = []

for env_type in EnvironmentType:

    # CRL configurations where the combination strategy is always RLforCD, CD, Repeat and ModelDiscoveryStrategy is to perform allways the same action, also each episode start in a different consecutive state to cover all the cases
    crl_conf_go = CRLConf("CRL-T{}_GO".format(T), combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.RANDOM, 0, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_gu = CRLConf("CRL-T{}-GU".format(T), combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.GU, 0, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_bc = CRLConf("CRL-T{}-BC".format(T), combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.BC, 0, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_dc = CRLConf("CRL-T{}-DC".format(T), combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.DC, 0, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    # One time run experiment to make a full exploration to discover the underlying Ground Truths for each action
    experiment = ExpConf("Ground Truth search", EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 5000, 10, ActionCountStrategy.Original, False, [crl_conf_go, crl_conf_gu, crl_conf_bc, crl_conf_dc])
    exp_coffee_0.append(experiment)


#############################################################################################
# FIRST experiment to test the advances of our proposed new RLforCD stage against RL        #
#############################################################################################
# The CRL configurations used changes the T parameter, the ModelUseStrategy for RLusingCD stages and the ModelDiscoveryStrategy for RLforCD stages

TRIALS = 1
T_VALUES = [10, 20, 30, 60]
MIN_FREQUENCY = 20
exp_coffee_1 = []

for env_type in EnvironmentType:
    # The RL configuration depends on the environment type to set the alpha value accordingly,
    # but in this experiment we allways start in 1.0epsilon and decrease until 0.1
    # That is the reason we always pick the index 2

    if env_type == EnvironmentType.DETERMINISTIC:
        rl_params = RL_DETERMINISTIC_CONFS[2]
        crl_params = RL_DETERMINISTIC_CONFS[2]
    elif env_type == EnvironmentType.STOCHASTIC:
        rl_params = RL_STOCHASTIC_CONFS[2]
        crl_params = RL_STOCHASTIC_CONFS[2]

    for T in T_VALUES:
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
        experiment = ExpConf("RL vs RLforCD T{}".format(T), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf, carl_conf])
        exp_coffee_1.append(experiment)


######################################################################################################################
# SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL  #
######################################################################################################################
# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings

TRIALS = 1
MIN_FREQUENCY = 20
exp_coffee_2 = []
T = 60

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
        experiment = ExpConf("CARL T{} vs RL e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_coffee_2.append(experiment)


######################################################################
# Third experiment to show what happens under different values of T  #
######################################################################

TRIALS = 1
T_VALUES = [10, 20, 40, 60]
MIN_FREQUENCY = 20
exp_coffee_3 = []

for env_type in EnvironmentType:
    # The RL configuration depends on the environment type to set the alpha value accordingly,
    # but in this experiment we allways start in 1.0epsilon and decrease until 0.1
    # That is the reason we always pick the index 2

    if env_type == EnvironmentType.DETERMINISTIC:
        rl_params = RL_DETERMINISTIC_CONFS[2]
        crl_params = RL_DETERMINISTIC_CONFS[2]
    elif env_type == EnvironmentType.STOCHASTIC:
        rl_params = RL_STOCHASTIC_CONFS[2]
        crl_params = RL_STOCHASTIC_CONFS[2]

    for T in T_VALUES:

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} vs RL e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_coffee_3.append(experiment)


#######################################################################################################################
# Four experiment to show what happens if with just discover a model once and then use it for the remaining episodes  #
#######################################################################################################################

TRIALS = 1
MIN_FREQUENCY = 20
exp_coffee_4 = []
T = 60

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} discover once use forever e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_coffee_4.append(experiment)



#############################################################
#                    TAXI SMALL TASK                        #
#############################################################

#############################################################
# ZERO experiment to find the Ground Truth Models           #
#############################################################

TRIALS = 1
T = 1000
exp_taxi_small_0_0 = []
MIN_FREQUENCY = 20

for env_type in EnvironmentType:

    # CRL configurations where the combination strategy is always RLforCD, CD, Repeat and ModelDiscoveryStrategy is to perform allways the same action, also each episode start in a different consecutive state to cover all the cases
    crl_conf_south = CRLConf("CRL-T60_SOUTH", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.SOUTH, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_north = CRLConf("CRL-T60-NORTH", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.NORTH, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_east = CRLConf("CRL-T60-EAST", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EAST, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_west = CRLConf("CRL-T60-WEST", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.WEST, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_pick = CRLConf("CRL-T60-PICK", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.PICK, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    crl_conf_drop = CRLConf("CRL-T60-DROP", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.DROP, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

    # One time run experiment to make a full exploration to discover the underlying Ground Truths for each action
    experiment = ExpConf("Ground Truth search", EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 100000, 1, ActionCountStrategy.Original, False, [crl_conf_drop, crl_conf_pick, crl_conf_south, crl_conf_north, crl_conf_east, crl_conf_west])
    exp_taxi_small_0_0.append(experiment)

#############################################################
# ZERO experiment to find the best Model-free RL parameters #
#############################################################

TRIALS = 1
MIN_FREQUENCY = 20
exp_taxi_small_0_1 = []
T = 60

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    # We test for different epsilons from 1 to 0.1
    experiment = ExpConf("Find best Model-free RL params", EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 100, ActionCountStrategy.Relational, True, RL_CONFS)
    exp_taxi_small_0_1.append(experiment)

    # Given the best parameter in the model free setting we test at different values of N for Dyna-Q
    # exp_taxi_small_0_2 = ExpConf("Find best Model-based RL params", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 2, 10, EvaluationMetric.EPISODE_REWARD, 1000, 100, ActionCountStrategy.Relational, True, [rl_conf_5, rl_conf_6, rl_conf_7, rl_conf_8])


#############################################################################################
# FIRST experiment to test the advances of our proposed new RLforCD, CD stage against RL,CD #
#############################################################################################
# The CRL configurations used changes the T parameter, the ModelUseStrategy for RLusingCD stages and the ModelDiscoveryStrategy for RLforCD stages

TRIALS = 1
T_VALUES = [10, 20, 50, 100]
MIN_FREQUENCY = 20
exp_taxi_small_1 = []

for env_type in EnvironmentType:
    # The RL configuration depends on the environment type to set the alpha value accordingly,
    # but in this experiment we allways start in 1.0epsilon and decrease until 0.1
    # That is the reason we always pick the index 2

    if env_type == EnvironmentType.DETERMINISTIC:
        rl_params = RL_DETERMINISTIC_CONFS[2]
        crl_params = RL_DETERMINISTIC_CONFS[2]
    elif env_type == EnvironmentType.STOCHASTIC:
        rl_params = RL_STOCHASTIC_CONFS[2]
        crl_params = RL_STOCHASTIC_CONFS[2]

    for T in T_VALUES:
        # The CRL configurations used just changes the T parameter
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
        experiment = ExpConf("RL vs RLforCD T{}".format(T), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf, carl_conf])
        exp_taxi_small_1.append(experiment)


#####################################################################################################################
# SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL #
#####################################################################################################################
# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings
TRIALS = 1
MIN_FREQUENCY = 20
exp_coffee_2 = []
T = 50
exp_taxi_small_2 = []

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
        experiment = ExpConf("CARL T{} vs RL e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_taxi_small_2.append(experiment)


######################################################################
# Third experiment to show what happens under different values of T  #
######################################################################

TRIALS = 1
T_VALUES = [10, 20, 100, 150]
MIN_FREQUENCY = 20
exp_taxi_small_3 = []

for env_type in EnvironmentType:
    # The RL configuration depends on the environment type to set the alpha value accordingly,
    # but in this experiment we allways start in 1.0epsilon and decrease until 0.1
    # That is the reason we always pick the index 2

    if env_type == EnvironmentType.DETERMINISTIC:
        rl_params = RL_DETERMINISTIC_CONFS[2]
        crl_params = RL_DETERMINISTIC_CONFS[2]
    elif env_type == EnvironmentType.STOCHASTIC:
        rl_params = RL_STOCHASTIC_CONFS[2]
        crl_params = RL_STOCHASTIC_CONFS[2]

    for T in T_VALUES:

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} vs RL e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_taxi_small_3.append(experiment)


# ######################################################################################################################
# # Four experiment to show what happens if with just discover a model once and then use it for the remaining episodes #
# ######################################################################################################################

TRIALS = 1
MIN_FREQUENCY = 20
exp_taxi_small_4 = []
T = 50

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} discover once use forever e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_taxi_small_4.append(experiment)


#############################################################
#                     TAXI BIG TASK                         #
#############################################################

# #############################################################################################################################################
# # First experiment to test the transfer learning capabilities of our model form TaxiSmall to TaxiBig using models at diffrent levels of SHD #
# #############################################################################################################################################
# The hypothesis is that our method perform better no matter how good is the model and no matter the epsilon.
# We do not compare here against Dyna-Q

TRIALS = 1
MIN_FREQUENCY = 20
exp_taxi_big_1 = []
T = 2500

MODELS_PATH = {EnvironmentType.DETERMINISTIC: {
                   "25":  "1-TaxiTask/20230605 130343 TaxiSmallEnv-deterministic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_99/10/",
                   "50":  "1-TaxiTask/20230605 130343 TaxiSmallEnv-deterministic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_95/60/",
                   "75":  "1-TaxiTask/20230605 130343 TaxiSmallEnv-deterministic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_73/300/",
                   "100": "1-TaxiTask/20230605 130343 TaxiSmallEnv-deterministic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_19/900/"},
               EnvironmentType.STOCHASTIC: {
                   "25": "1-TaxiTask/20230605 125830 TaxiSmallEnv-stochastic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_99/10/",
                   "50": "1-TaxiTask/20230605 125830 TaxiSmallEnv-stochastic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_91/100/",
                   "75": "1-TaxiTask/20230605 125830 TaxiSmallEnv-stochastic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_67/370/",
                   "100": "1-TaxiTask/20230605 125830 TaxiSmallEnv-stochastic RL vs RLforCD T10 Epi = 1000 steps = 50 ace = relational sis = True trials 1/trial 1/cd_data_and_results/CARL agent/0_11/990/"}
               }

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        rl_params = RL_DETERMINISTIC_CONFS[2]
        crl_params = RL_DETERMINISTIC_CONFS[2]
    elif env_type == EnvironmentType.STOCHASTIC:
        rl_params = RL_STOCHASTIC_CONFS[2]
        crl_params = RL_STOCHASTIC_CONFS[2]

    model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)

    # 100% Model
    carl_conf_100 = CRLConf("CARL-100%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, MODELS_PATH[env_type]["100"])
    # 75% Model
    carl_conf_75 = CRLConf("CARL-75%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, MODELS_PATH[env_type]["75"])
    # 50% Model
    carl_conf_50 = CRLConf("CARL-50%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, MODELS_PATH[env_type]["50"])
    # 25% Model
    carl_conf_25 = CRLConf("CARL-25%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                         T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, MODELS_PATH[env_type]["25"])

    experiment = ExpConf("Taxi big task CARL vs RL using partial models", EnvironmentNames.TAXI_BIG, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 2000, 100, ActionCountStrategy.Original, True, [carl_conf_100, carl_conf_75, carl_conf_50, carl_conf_25, model_free_rl])
    exp_taxi_big_1.append(experiment)


