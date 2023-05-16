# TODO Create a dic here to place all the parameters for each task
from abc import abstractmethod
from enum import Enum
import numpy as np

class EnvironmentNames(Enum):
    COFFEE = "CoffeeTaskEnv-v0"
    TAXI_SMALL = "TaxiSmallEnv-v0"
    TAXI_BIG = "TaxiBigEnv-v0"

class EnvironmentType(Enum):
    STOCHASTIC = "stochastic"
    DETERMINISTIC = "deterministic"

class ActionCountStrategy(Enum):
    Relational = "relational"
    Original = "original"

class EpisodeStateInitialization(Enum):
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
    def __init__(self, alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization, T, th, model_use_strategy, model_discovery_strategy, crl_action_selection_strategy, use_crl_data, model_init_path = None):
        super().__init__(alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization)
        self.T = T
        self.th = th
        self.model_use_strategy = model_use_strategy
        self.model_discovery_strategy = model_discovery_strategy
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

cycle_stage_0 = Stage([Step.RL, Step.CD, Step.RL_USING_CD], Times.FOREVER)
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

# DETERMINISTIC ENVIRONMENT WITH DECAYED EPSILON
rl_conf_1 = RLConf("Q-Learning e0.3dec", combination_strategy_7, 0, 1, 0.95, 0.3, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_2 = RLConf("Q-Learning e0.7dec", combination_strategy_7, 0, 1, 0.95, 0.7, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_3 = RLConf("Q-Learning e1.0dec", combination_strategy_7, 0, 1, 0.95, 1.0, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_4 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)

# DETERMINISTIC ENVIRONMENT WITH FIXED EPSILON
rl_conf_5 = RLConf("Dyna Q n5", combination_strategy_7, 5, 1, 0.95, 0.1, 0.1,  ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_6 = RLConf("Dyna Q n10", combination_strategy_7, 10, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_7 = RLConf("Dyna Q n20", combination_strategy_7, 20, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_8 = RLConf("Dyna Q n50", combination_strategy_7, 50, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)


#############################################################
#               COFFEE TASK (DETERMINISTIC)                 #
#############################################################

#############################################################
# ZERO experiment to find the Ground Truth Models           #
#############################################################

# CRL configurations where the combination strategy is always RLforCD, CD, Repeat and ModelDiscoveryStrategy is to perform allways the same action, also each episode start in a different consecutive state to cover all the cases
crl_conf_go = CRLConf("CRL-T60_GO", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                     60, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.GO, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

crl_conf_gu = CRLConf("CRL-T60-GU", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                     60, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.GU, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

crl_conf_bc = CRLConf("CRL-T60-BC", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                     60, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.BC, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

crl_conf_dc = CRLConf("CRL-T60-DC", combination_strategy_6, rl_conf_1.n, rl_conf_1.alpha, rl_conf_1.gamma, rl_conf_1.epsilon_start, rl_conf_1.epsilon_end,  rl_conf_1.rl_action_selection_strategy, EpisodeStateInitialization.EPISODE_NUMBER,
                     60, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.DC, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# One time run experiment to make a full exploration to discover the underlying Ground Truths for each action
exp_coffee_0 = ExpConf("Ground Truth search", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 100, ActionCountStrategy.Original, True, [crl_conf_go, crl_conf_gu, crl_conf_bc, crl_conf_dc])

#############################################################################################
# FIRST experiment to test the advances of our proposed new RLforCD, CD stage against RL,CD #
#############################################################################################

# The RL configuration is the deterministic to start in 1.0 epsilon and decrease until 0.1
rl_agent_params = rl_conf_3
rl_for_cd_agent_params = rl_conf_3

# The CRL configurations used just changes the T parameter in 10,20,50,100
crl_conf_100 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_101 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)


# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_1_1 = ExpConf("RL vs RL for CD T10", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [crl_conf_100, crl_conf_101])

crl_conf_102 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_103 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_1_2 = ExpConf("RL vs RL for CD T20", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [crl_conf_102, crl_conf_103])

crl_conf_104 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_105 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_1_3 = ExpConf("RL vs RL for CD T50", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [crl_conf_104, crl_conf_105])

crl_conf_106 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_107 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_1_4 = ExpConf("RL vs RL for CD T100", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [crl_conf_106, crl_conf_107])


# #####################################################################################################################
# # SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL #
# #####################################################################################################################

# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings
rl_params = rl_conf_1
crl_params = rl_conf_1

rl_conf_18 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_19 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_1 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_12 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_2_1 = ExpConf("CARL T50 vs RL e0.3dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf_1, crl_conf_12])


rl_params = rl_conf_2
crl_params = rl_conf_2

rl_conf_20 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_21 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_2 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_13 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_2_2 = ExpConf("CARL T50 vs RL e0.7dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf_2, crl_conf_13])


rl_params = rl_conf_3
crl_params = rl_conf_3
rl_conf_22 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_23 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_3 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_14 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_2_3 = ExpConf("CARL T50 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf_3, crl_conf_14])


rl_params = rl_conf_4
crl_params = rl_conf_4
rl_conf_24 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_25 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_4 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_15 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_coffee_2_4 = ExpConf("CARL T50 vs RL e0.1dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf_4, crl_conf_15])

# #####################################################################
# # Third experiment to show what happens under different values of T #
# #####################################################################
#
# # For this experiment we use epsilon 1.0 decayed
rl_params = rl_conf_3
crl_params = rl_conf_3
rl_conf_22 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_23 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_14 = CRLConf("CARL T10", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_15 = CRLConf("CARL T20", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_16 = CRLConf("CARL T100", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_17 = CRLConf("CRL T150", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     150, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_coffee_3_1 = ExpConf("CARL T10 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_22, rl_conf_23, crl_conf_14])

exp_coffee_3_2 = ExpConf("CARL T20 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True,[rl_conf_22, rl_conf_23, crl_conf_15])

exp_coffee_3_3 = ExpConf("CARL T100 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True,[rl_conf_22, rl_conf_23, crl_conf_16])

exp_coffee_3_4 = ExpConf("CARL T150 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_22, rl_conf_23, crl_conf_17])


# ######################################################################################################################
# # Four experiment to show what happens if with just discover a model once and then use it for the remaining episodes #
# ######################################################################################################################

rl_params = rl_conf_1
crl_params = rl_conf_1

rl_conf_23 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_24 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_18 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_coffee_4_1 = ExpConf("CARL T50 vs RL e0.3dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_23, rl_conf_24, crl_conf_18])


rl_params = rl_conf_2
crl_params = rl_conf_2

rl_conf_25 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_26 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_19 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_coffee_4_2 = ExpConf("CARL T50 vs RL e0.7dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_25, rl_conf_26, crl_conf_19])


rl_params = rl_conf_3
crl_params = rl_conf_3

rl_conf_27 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_28 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_20 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_coffee_4_3 = ExpConf("CARL T50 vs RL e1.0dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_27, rl_conf_28, crl_conf_20])


rl_params = rl_conf_4
crl_params = rl_conf_4

rl_conf_29 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_30 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_21 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_coffee_4_4 = ExpConf("CARL T50 vs RL e0.1dec", EnvironmentNames.COFFEE, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [rl_conf_29, rl_conf_30, crl_conf_21])



#############################################################
#                 TAXI TASK (DETERMINISTIC)                 #
#############################################################

#############################################################
# ZERO experiment to find the best Model-free RL parameters #
#############################################################

# In this experiment we fix alpha to 1 that is optimal in deterministic environments, and we also fix gamma to .95
# We use different epsilons from 1 to 0.1
exp_taxi_small_0_1 = ExpConf("Find best Model-free RL params", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 100, ActionCountStrategy.Relational, True, [rl_conf_1, rl_conf_2, rl_conf_3, rl_conf_4])
# SEE RESULTS at 20230206 172941

# ZERO experiment to find the best Model-based RL parameters
# Given the best parameter in the model free setting we test at different values of N for Dyna-Q
exp_taxi_small_0_2 = ExpConf("Find best Model-based RL params", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 2, 10, EvaluationMetric.EPISODE_REWARD, 1000, 100, ActionCountStrategy.Relational, True, [rl_conf_5, rl_conf_6, rl_conf_7, rl_conf_8])
# SEE RESULTS at 20230206 173517


#############################################################################################
# FIRST experiment to test the advances of our proposed new RLforCD, CD stage against RL,CD #
#############################################################################################
### IDEA de Eduardo

# The RL configuration is the deterministic to start in 1.0 epsilon and decrease until 0.1
rl_agent_params = rl_conf_3
rl_for_cd_agent_params = rl_conf_3

# The CRL configurations used just changes the T parameter
crl_conf_100 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_101 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_taxi_small_1_1 = ExpConf("RL vs RL for CD T10", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [crl_conf_100, crl_conf_101])

crl_conf_102 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_103 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
exp_taxi_small_1_2 = ExpConf("RL vs RL for CD T20", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [crl_conf_102, crl_conf_103])

crl_conf_104 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_105 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
exp_taxi_small_1_3 = ExpConf("RL vs RL for CD T50", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [crl_conf_104, crl_conf_105])

crl_conf_106 = CRLConf("RL agent", combination_strategy_0, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_107 = CRLConf("RL for CD agent", combination_strategy_1, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
exp_taxi_small_1_4 = ExpConf("RL vs RL for CD T100", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [crl_conf_106, crl_conf_107])

# #####################################################################################################################
# # SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL #
# #####################################################################################################################
#
# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings
rl_params = rl_conf_1
crl_params = rl_conf_1

rl_conf_18 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_19 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_1 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_12 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_taxi_small_2_1 = ExpConf("CARL T50 vs RL e0.3dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf_1, crl_conf_12])


rl_params = rl_conf_2
crl_params = rl_conf_2

rl_conf_20 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_21 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_2 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_13 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_taxi_small_2_2 = ExpConf("CARL T50 vs RL e0.7dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf_2, crl_conf_13])


rl_params = rl_conf_3
crl_params = rl_conf_3
rl_conf_22 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_23 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_3 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_14 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_taxi_small_2_3 = ExpConf("CARL T50 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf_3, crl_conf_14])


rl_params = rl_conf_4
crl_params = rl_conf_4
rl_conf_24 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_25 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
pgm_conf_4 = CRLConf("PGM agent", combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_15 = CRLConf("CARL T50", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
exp_taxi_small_2_4 = ExpConf("CARL T50 vs RL e0.1dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf_4, crl_conf_15])



# #####################################################################
# # Third experiment to show what happens under different values of T #
# #####################################################################
#
# # For this experiment we use epsilon 1.0 decayed
rl_params = rl_conf_3
crl_params = rl_conf_3
rl_conf_22 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_23 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_14 = CRLConf("CARL T10", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     10, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_15 = CRLConf("CARL T20", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     20, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_16 = CRLConf("CARL T100", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     100, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
crl_conf_17 = CRLConf("CRL T150", combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     150, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

#exp_taxi_small_3_0 = ExpConf("Taxi small task RL vs CRL e1.0dec different T all together", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 200, ActionCountStrategy.Relational, True, [rl_conf_22, rl_conf_23, crl_conf_14, crl_conf_15, crl_conf_16, crl_conf_17])


exp_taxi_small_3_1 = ExpConf("CARL T10 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_22, rl_conf_23, crl_conf_14])


exp_taxi_small_3_2 = ExpConf("CARL T20 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True,[rl_conf_22, rl_conf_23, crl_conf_15])


exp_taxi_small_3_3 = ExpConf("CARL T100 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True,[rl_conf_22, rl_conf_23, crl_conf_16])


exp_taxi_small_3_4 = ExpConf("CARL T150 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_22, rl_conf_23, crl_conf_17])


# ######################################################################################################################
# # Four experiment to show what happens if with just discover a model once and then use it for the remaining episodes #
# ######################################################################################################################


rl_params = rl_conf_1
crl_params = rl_conf_1

rl_conf_23 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_24 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_18 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_taxi_small_4_1 = ExpConf("CARL T50 vs RL e0.3dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_23, rl_conf_24, crl_conf_18])


rl_params = rl_conf_2
crl_params = rl_conf_2

rl_conf_25 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_26 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_19 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_taxi_small_4_2 = ExpConf("CARL T50 vs RL e0.7dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_25, rl_conf_26, crl_conf_19])


rl_params = rl_conf_3
crl_params = rl_conf_3

rl_conf_27 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_28 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_20 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_taxi_small_4_3 = ExpConf("CARL T50 vs RL e1.0dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_27, rl_conf_28, crl_conf_20])


rl_params = rl_conf_4
crl_params = rl_conf_4

rl_conf_29 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_30 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
crl_conf_21 = CRLConf("CARL T50", combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

exp_taxi_small_4_4 = ExpConf("CARL T50 vs RL e0.1dec", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [rl_conf_29, rl_conf_30, crl_conf_21])

#
#
#
# # # Deterministic TaxiBig Task
#
# #############################################################################################################################################
# # First experiment to test the transfer learning capabilities of our model form TaxiSmall to TaxiBig using models at diffrent levels of SHD #
# #############################################################################################################################################
#
# The hypothesis is that our method perform better no matter how good is the model and no matter the epsilon.
# We do not compare here against Dyna-Q

rl_params = rl_conf_3
crl_params = rl_conf_3

rl_conf_31 = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
rl_conf_32 = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end,  rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
# FULL Model
crl_conf_22 = CRLConf("CARL-100%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, "experiments_results/20230201 163520 Taxi small task RL vs CRL TaxiSmallEnv Episodes = 1000 trials 1/trial 1/cd_data_and_results/CRL-T100/0_41/900/")

# For partial models use the ones on 20230203 125808 TaxiSmallEnv RL vs RLforCD Episodes = 600 trials 10

# 75% Model
crl_conf_23 = CRLConf("CARL-75%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, "experiments_results/20230203 125808 TaxiSmallEnv RL vs RLforCD Episodes = 600 trials 10/trial 1/cd_data_and_results/RLforCD Agent/0_98/20/")

# 50% Model
crl_conf_24 = CRLConf("CARL-50%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end,  crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, "experiments_results/20230203 125808 TaxiSmallEnv RL vs RLforCD Episodes = 600 trials 10/trial 1/cd_data_and_results/RLforCD Agent/0_95/50/")

# 25% Model
crl_conf_25 = CRLConf("CARL-25%", combination_strategy_4, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                     50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, "experiments_results/20230203 125808 TaxiSmallEnv RL vs RLforCD Episodes = 600 trials 10/trial 1/cd_data_and_results/RLforCD Agent/1_00/5/")


exp_taxi_big_1_1 = ExpConf("Taxi big task CARL vs RL using partial models", EnvironmentNames.TAXI_BIG, EnvironmentType.DETERMINISTIC, 10, 10, EvaluationMetric.EPISODE_REWARD, 2500, 100, ActionCountStrategy.Relational, True, [crl_conf_22, crl_conf_23, crl_conf_24, crl_conf_25, rl_conf_31])



# ###################################################################################################
# Original idea to measure the advantages of our RlforCD over traditional RL for causal discovery   #
# ###################################################################################################

# We test an agent doing RLforCD against RL only at different exploration levels
# We use small T value of 5 to have more data

#rl_conf_14 = RLConf("Q-Learning e0.3fix", combination_strategy_7, 0, 1, 0.95, 0.3, 0.3, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
#rl_conf_15 = RLConf("Q-Learning e0.7fix", combination_strategy_7, 0, 1, 0.95, 0.7, 0.7, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
#rl_conf_16 = RLConf("Q-Learning e1.0fix", combination_strategy_7, 0, 1, 0.95, 1.0, 1.0, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
#rl_conf_17 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)

# crl_conf_98 = CRLConf("RLforCD only", combination_strategy_6, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
#
# crl_conf_99 = CRLConf("RL only", combination_strategy_5, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      50, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

# rl_agent_params = rl_conf_14
# rl_for_cd_agent_params = rl_conf_14
#
# # CRL Conf (alg_name, combination_strategy, n, alpha, gamma, .epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization,
# #           T, th, model_use_strategy, model_discovery_strategy, crl_action_selection_strategy, use_crl_data, model_init_path = None)
# crl_conf_4 = CRLConf("epsilon-greedy agent", combination_strategy_5, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
#
# crl_conf_5 = CRLConf("RL for CD agent", combination_strategy_6, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.NEW_IDEA, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
# # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
# exp_taxi_small_1_1 = ExpConf("RL vs RL for CD e0.3fix", EnvironmentNames.TAXI_BIG, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 500, 100, [crl_conf_4, crl_conf_5])
# 
#
# rl_agent_params = rl_conf_15
# rl_for_cd_agent_params = rl_conf_15
# crl_conf_6 = CRLConf("RL Agent", combination_strategy_5, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
#
# crl_conf_7 = CRLConf("RL for CD Agent", combination_strategy_6, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
# exp_taxi_small_1_2 = ExpConf("RL vs RL for CD e0.7fix", EnvironmentNames.TAXI_BIG, EnvironmentType.DETERMINISTIC, 5, 10, EvaluationMetric.EPISODE_REWARD, 500, 100, [crl_conf_6, crl_conf_7])
# 
#
# rl_agent_params = rl_conf_3
# rl_for_cd_agent_params = rl_conf_3
# crl_conf_8 = CRLConf("RL Agent", combination_strategy_5, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
#
# crl_conf_9 = CRLConf("RLforCD Agent", combination_strategy_6, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
#                      5, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
# exp_taxi_small_1_3 = ExpConf("RL vs RLforCD e1.0fix", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 500, 50, [crl_conf_8, crl_conf_9])
# 
#
# rl_agent_params = rl_conf_17
# rl_for_cd_agent_params = rl_conf_17
# crl_conf_10 = CRLConf("RL Agent", combination_strategy_5, rl_agent_params.n, rl_agent_params.alpha, rl_agent_params.gamma, rl_agent_params.epsilon_start, rl_agent_params.epsilon_end, rl_agent_params.rl_action_selection_strategy, rl_agent_params.episode_state_initialization,
#                      1, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
#
# crl_conf_11 = CRLConf("RLforCD Agent", combination_strategy_6, rl_for_cd_agent_params.n, rl_for_cd_agent_params.alpha, rl_for_cd_agent_params.gamma, rl_for_cd_agent_params.epsilon_start, rl_for_cd_agent_params.epsilon_end, rl_for_cd_agent_params.rl_action_selection_strategy, rl_for_cd_agent_params.episode_state_initialization,
#                      1, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
# exp_taxi_small_1_4 = ExpConf("RL vs RLforCD e0.1fix", EnvironmentNames.TAXI_SMALL, EnvironmentType.DETERMINISTIC, 1, 10, EvaluationMetric.EPISODE_REWARD, 1000, 100, [crl_conf_10, crl_conf_11])
# 
