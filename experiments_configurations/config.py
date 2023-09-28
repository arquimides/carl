from enum import Enum

class EnvironmentNames(Enum):
    COFFEE = "our_gym_environments/CoffeeTaskEnv-v0"
    TAXI_SMALL = "our_gym_environments/TaxiSmallEnv-v0"
    TAXI_BIG = "our_gym_environments/TaxiBigEnv-v0"
    TAXI_ATARI_SMALL = "our_gym_environments/TaxiAtariSmallEnv-v0"

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


class DeepRLConf(AlgConf):
    def __init__(self, alg_name, combination_strategy, screen_width=84, screen_height=84,
                 learning_rate=1e-4, buffer_size=500000, gamma=0.99, target_network_update_rate=1.,
                 target_network_update_frequency=10000, batch_size=32, start_e=1.0, end_e=0.01,
                 exploration_fraction=0.10,
                 learning_start=80000, train_frequency=4):

        super().__init__(alg_name, combination_strategy)

        self.screen_width = screen_width  # Width of the game screen
        self.screen_height = screen_height  # Height of the game screen
        self.learning_rate = learning_rate  # The learning rate of the algorithm
        self.buffer_size = buffer_size  # The size of the replay memory buffer
        self.gamma = gamma  # The discount factor gamma
        self.target_network_update_rate = target_network_update_rate  # The target network update rate
        self.target_network_update_frequency = target_network_update_frequency  # The frequency of target network updates
        self.batch_size = batch_size  # The batch size for training
        self.start_e = start_e  # The starting epsilon for exploration
        self.end_e = end_e  # The ending epsilon for exploration
        self.exploration_fraction = exploration_fraction  # The fraction of total time steps for epsilon decay
        self.learning_start = learning_start  # The time step to start learning
        self.train_frequency = train_frequency  # The frequency of training



class DeepCRLConf(DeepRLConf):
    def __init__(self, alg_name, combination_strategy, screen_width=84, screen_height=84,
                 learning_rate=1e-4, buffer_size=500000, gamma=0.99, target_network_update_rate=1.,
                 target_network_update_frequency=10000, batch_size=32, start_e=1.0, end_e=0.01,
                 exploration_fraction=0.10, learning_start=80000, train_frequency=4,
                 T=30000, th=0.7, min_frequency=30, model_use_strategy=None, model_discovery_strategy=None,
                 crl_action_selection_strategy=None, use_crl_data=True, model_init_path=None):

        super().__init__(alg_name, combination_strategy, screen_width, screen_height,
                         learning_rate, buffer_size, gamma, target_network_update_rate,
                         target_network_update_frequency, batch_size, start_e, end_e,
                         exploration_fraction, learning_start, train_frequency)

        # CARL related
        self.T = T
        self.th = th
        self.min_frequency = min_frequency
        self.model_use_strategy = model_use_strategy
        self.model_discovery_strategy = model_discovery_strategy
        self.crl_action_selection_strategy = crl_action_selection_strategy
        self.use_crl_data = use_crl_data
        self.model_init_path = model_init_path


class RLConf(AlgConf):
    def __init__(self, alg_name, combination_strategy, n, alpha, gamma, epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization):
        super().__init__(alg_name, combination_strategy)
        self.n = n
        self.alpha = alpha  # learning rate (Use 1 as optimal for deterministic environments, use 0.1 for stochastic)
        self.gamma = gamma  # discount rate
        self.epsilon_start = epsilon_start  # Exploration factor (1 is full exploration, 0 is none exploration)
        self.epsilon_end = epsilon_end  # Exploration factor (1 is full exploration, 0 is none exploration)
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

cycle_stage_4 = Stage([Step.RL_USING_CD, Step.CD], Times.FOREVER)
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
combination_strategy_9 = CombinationStrategy("RL and then CD and RL using CD forever", [single_stage_1, cycle_stage_9])

#############################################################
#                  RL Configurations                        #
#############################################################

# RL Configurations (alg_name, combination_strategy, n, alpha, gamma, .epsilon_start, epsilon_end, rl_action_selection_strategy, episode_state_initialization):

# DETERMINISTIC ENVIRONMENT WITH DECAYED EPSILON, alpha = 1.0 (optimal)
rl_conf_1 = RLConf("Q-Learning e0.3dec", combination_strategy_7, 0, 1, 0.95, 0.3, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_2 = RLConf("Q-Learning e0.7dec", combination_strategy_7, 0, 1, 0.95, 0.7, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_3 = RLConf("Q-Learning e1.0dec", combination_strategy_7, 0, 1, 0.95, 1.0, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_4 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 1, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)

# STOCHASTIC ENVIRONMENT WITH DECAYED EPSILON, alpha = 0.3
rl_conf_5 = RLConf("Q-Learning e0.3dec", combination_strategy_7, 0, 0.3, 0.95, 0.3, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_6 = RLConf("Q-Learning e0.7dec", combination_strategy_7, 0, 0.3, 0.95, 0.7, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_7 = RLConf("Q-Learning e1.0dec", combination_strategy_7, 0, 0.3, 0.95, 1.0, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)
rl_conf_8 = RLConf("Q-Learning e0.1fix", combination_strategy_7, 0, 0.3, 0.95, 0.1, 0.1, ActionSelectionStrategy.EPSILON_GREEDY, EpisodeStateInitialization.RANDOM)


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

TRIALS = 20
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
        experiment = ExpConf("RL vs RLforCD T{} th{}".format(T, 0.7), EnvironmentNames.COFFEE, env_type, TRIALS, 1, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf, carl_conf])
        exp_coffee_1.append(experiment)


######################################################################################################################
# SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL  #
######################################################################################################################
# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings

TRIALS = 20
MIN_FREQUENCY = 20
exp_coffee_2 = []
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
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
        experiment = ExpConf("CARL T{} vs RL e{}dec th{}".format(T, crl_params.epsilon_start, 0.7), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_coffee_2.append(experiment)


######################################################################
# Third experiment to show what happens under different values of T  #
######################################################################

TRIALS = 20
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

TRIALS = 20
MIN_FREQUENCY = 20
exp_coffee_4 = []
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
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_9, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} discover once use forever e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_coffee_4.append(experiment)

#######################################################################################################################
# FIVE EXPERIMENT  #
#######################################################################################################################

TRIALS = 20
MIN_FREQUENCY = 20
exp_coffee_5 = []
T = 50

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        carl_conf_1 = CRLConf("CARL T{} C_1".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf_2 = CRLConf("CARL T{} C_2".format(T), combination_strategy_9, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} using C1 vs C2 e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [carl_conf_1, carl_conf_2])
        exp_coffee_5.append(experiment)



#############################################################
#                    TAXI SMALL TASK                        #
#############################################################

#############################################################
# ZERO experiment to find the Ground Truth Models           #
#############################################################

TRIALS = 20
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

TRIALS = 20
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

TRIALS = 20
T_VALUES = [10, 20, 50, 100]
MIN_FREQUENCY = 30
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
        experiment = ExpConf("RL vs RLforCD T{} th{}".format(T,0.7), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf, carl_conf])
        exp_taxi_small_1.append(experiment)


#####################################################################################################################
# SECOND experiment to show the advances of our proposed combination strategy against model free and model based RL #
#####################################################################################################################
# We compare our method against different Model-free and Model-based params. We want to test how our method performs in police learning and causal discovery for each settings
TRIALS = 20
MIN_FREQUENCY = 20
T = 50
exp_taxi_small_2 = []

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        for th in [0.01]:
            rl_params = RL_CONFS[i]
            crl_params = RL_CONFS[i]

            #model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
            #                     T, th, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("CARL T{} vs RL e{}dec th{}".format(T, crl_params.epsilon_start, th), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [carl_conf])
            exp_taxi_small_2.append(experiment)


######################################################################
# Third experiment to show what happens under different values of T  #
######################################################################

TRIALS = 20
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

TRIALS = 20
MIN_FREQUENCY = 20
exp_taxi_small_4 = []
T = 200

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
        pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_9, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                           T, 0.7, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_8, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                             T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} discover once use forever e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 2000, 50, ActionCountStrategy.Relational, True, [model_free_rl, model_based_rl, pgm_conf, carl_conf])
        exp_taxi_small_4.append(experiment)

#######################################################################################################################
# FIVE EXPERIMENT  #
#######################################################################################################################

TRIALS = 20
MIN_FREQUENCY = 20
exp_taxi_small_5 = []
T = 50

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        rl_params = RL_CONFS[i]
        crl_params = RL_CONFS[i]

        carl_conf_1 = CRLConf("CARL T{} C_1".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                              T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
        carl_conf_2 = CRLConf("CARL T{} C_2".format(T), combination_strategy_9, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                              T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

        experiment = ExpConf("CARL T{} using C1 vs C2 e{}dec".format(T, crl_params.epsilon_start), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD,
                             1000, 50, ActionCountStrategy.Original, True, [carl_conf_1, carl_conf_2])
        exp_taxi_small_5.append(experiment)


#############################################################
#                     TAXI BIG TASK                         #
#############################################################

# #############################################################################################################################################
# # First experiment to test the transfer learning capabilities of our model form TaxiSmall to TaxiBig using models at diffrent levels of SHD #
# #############################################################################################################################################
# The hypothesis is that our method perform better no matter how good is the model and no matter the epsilon.
# We do not compare here against Dyna-Q

TRIALS = 20
MIN_FREQUENCY = 20
exp_taxi_big_1 = []
T = 50

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


#############################################################
#                SENSITIVE ANALYSIS - th param in CD        #
#############################################################
th_values = [0.01,0.1,0.3,0.5,0.7,0.9,0.99]

TRIALS = 20
T_VALUES = [10, 20, 30, 60]
MIN_FREQUENCY = 20
exp_coffee_extra_1 = []

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
        for th in th_values:
            pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("RL vs RLforCD T{} th{}".format(T, th), EnvironmentNames.COFFEE, env_type, TRIALS, 1, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf, carl_conf])
            exp_coffee_extra_1.append(experiment)

TRIALS = 10
T_VALUES = [50, 100]
MIN_FREQUENCY = 30
exp_taxi_small_extra_1 = []

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
        for th in th_values:
            # The CRL configurations used just changes the T parameter
            pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("RL vs RLforCD T{} th{}".format(T,th), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf, carl_conf])
            exp_taxi_small_extra_1.append(experiment)

#############################################################
#                SENSITIVE ANALYSIS - th param in PL        #
#############################################################

TRIALS = 20
MIN_FREQUENCY = 20
exp_coffee_extra_2 = []
T = 50

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        for th in th_values:
            rl_params = RL_CONFS[i]
            crl_params = RL_CONFS[i]

            model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
            #                     T, th, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("CARL T{} vs RL e{}dec th{}".format(T, crl_params.epsilon_start, th), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, carl_conf])
            exp_coffee_extra_2.append(experiment)

TRIALS = 10
MIN_FREQUENCY = 20
T = 50
exp_taxi_small_extra_2 = []

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        for th in th_values:
            rl_params = RL_CONFS[i]
            crl_params = RL_CONFS[i]

            model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
            #                     T, th, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, th, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("CARL T{} vs RL e{}dec th{}".format(T, crl_params.epsilon_start, th), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [model_free_rl, carl_conf])
            exp_taxi_small_extra_2.append(experiment)

#############################################################
#                SENSITIVE ANALYSIS - f param in CD         #
#############################################################

f_values = [1,5,10,20,50,100,500]

TRIALS = 20
T_VALUES = [10, 20, 30, 60]
exp_coffee_extra_3 = []

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
        for f in f_values:
            pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("RL vs RLforCD T{} f{}".format(T, f), EnvironmentNames.COFFEE, env_type, TRIALS, 1, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [pgm_conf, carl_conf])
            exp_coffee_extra_3.append(experiment)

TRIALS = 10
T_VALUES = [10, 20, 50, 100]
exp_taxi_small_extra_3 = []

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
        for f in f_values:
            # The CRL configurations used just changes the T parameter
            pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("RL vs RLforCD T{} f{}".format(T,f), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [pgm_conf, carl_conf])
            exp_taxi_small_extra_3.append(experiment)


#############################################################
#                SENSITIVE ANALYSIS - f param in PL         #
#############################################################

TRIALS = 20
exp_coffee_extra_4 = []
T = 50

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        for f in f_values:
            rl_params = RL_CONFS[i]
            crl_params = RL_CONFS[i]

            model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
            #                     T, th, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("CARL T{} vs RL e{}dec f{}".format(T, crl_params.epsilon_start, f), EnvironmentNames.COFFEE, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 600, 25, ActionCountStrategy.Original, True, [model_free_rl, carl_conf])
            exp_coffee_extra_4.append(experiment)

TRIALS = 10
T = 50
exp_taxi_small_extra_4 = []

for env_type in EnvironmentType:

    if env_type == EnvironmentType.DETERMINISTIC:
        RL_CONFS = RL_DETERMINISTIC_CONFS
    elif env_type == EnvironmentType.STOCHASTIC:
        RL_CONFS = RL_STOCHASTIC_CONFS

    for i in range(len(RL_CONFS)):
        for f in f_values:
            rl_params = RL_CONFS[i]
            crl_params = RL_CONFS[i]

            model_free_rl = RLConf("Q-Learning", combination_strategy_7, rl_params.n, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #model_based_rl = RLConf("Dyna Q n20", combination_strategy_7, 20, rl_params.alpha, rl_params.gamma, rl_params.epsilon_start, rl_params.epsilon_end, rl_params.rl_action_selection_strategy, rl_params.episode_state_initialization)
            #pgm_conf = CRLConf("PGM'22 T{}".format(T), combination_strategy_0, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
            #                     T, th, ModelUseStrategy.IMMEDIATE_POSITIVE, ModelDiscoveryStrategy.EPSILON_GREEDY, MIN_FREQUENCY, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)
            carl_conf = CRLConf("CARL T{}".format(T), combination_strategy_1, crl_params.n, crl_params.alpha, crl_params.gamma, crl_params.epsilon_start, crl_params.epsilon_end, crl_params.rl_action_selection_strategy, crl_params.episode_state_initialization,
                                 T, 0.7, ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE, ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY, f, ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, True, None)

            # ExpConf(exp_name, env_name, env_type, trials, smooth, evaluation_metric, max_episodes, max_steps, alg_confs):
            experiment = ExpConf("CARL T{} vs RL e{}dec f{}".format(T, crl_params.epsilon_start, f), EnvironmentNames.TAXI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 1000, 50, ActionCountStrategy.Relational, True, [model_free_rl, carl_conf])
            exp_taxi_small_extra_4.append(experiment)


#############################################################
#                DEEP-RL with CARL experiments              #
#############################################################
TRIALS = 1
exp_deep_rl_1 = []

for env_type in [EnvironmentType.DETERMINISTIC]:

    # Here we use the combination strategy RL for all episodes, algorithm param is 'dqn'
    dqn_conf = DeepRLConf("DQN", combination_strategy=combination_strategy_7, screen_width=84, screen_height=84,
                 learning_rate=1e-4, buffer_size=500000, gamma=0.99, target_network_update_rate=1.,
                 target_network_update_frequency=10000, batch_size=32, start_e=1.0, end_e=0.01, exploration_fraction=0.10,
                 learning_start=80000, train_frequency=4)

    # Here we use the combination strategy 'Discover once, use forever'
    carl_dqn_conf = DeepCRLConf("DQN", combination_strategy=combination_strategy_8, screen_width=84, screen_height=84,
                 learning_rate=1e-4, buffer_size=500000, gamma=0.99, target_network_update_rate=1.,
                 target_network_update_frequency=10000, batch_size=32, start_e=1.0, end_e=0.01, exploration_fraction=0.10,
                 learning_start=80000, train_frequency=4,
                 T=50000, th=0.7, min_frequency=30, model_use_strategy=ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE,
                 model_discovery_strategy=ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY,
                 crl_action_selection_strategy=ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY, use_crl_data=True, model_init_path=None)

    experiment = ExpConf("DQN vs CARL-DQN", EnvironmentNames.TAXI_ATARI_SMALL, env_type, TRIALS, 10, EvaluationMetric.EPISODE_REWARD, 15000000, 1000, ActionCountStrategy.Relational, True, [dqn_conf, carl_dqn_conf])

    exp_deep_rl_1.append(experiment)

