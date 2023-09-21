import gymnasium as gym
import our_gym_environments
import numpy as np
import rpy2.robjects as ro  # Packages for Python-R communication
import time

import util
import sys
from experiments_configurations import config
from experiments_configurations.config import EvaluationMetric, EpisodeStateInitialization, Times, \
    Step, ActionSelectionStrategy, ModelUseStrategy, ModelDiscoveryStrategy, CRLConf, ActionCountStrategy, \
    EnvironmentType, DeepCRLConf, DeepRLConf

# Configuring the rpy2 stuff for R communication
r = ro.r  # Creating the R instance
r['source']('causal_discovery.R')  # Loading and sourcing R file.
# Import R functions for later use
cd_function_r = ro.globalenv['causal_discovery_using_rl_data']
load_model_function_r = ro.globalenv['load_model']
dbn_inference_function_r = ro.globalenv['dbn_inference']
plot_gt_funtion_r = ro.globalenv['plot_ground_truths']

# Imports for Deep-RL
import argparse
import datetime
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import AveragedDQN, CategoricalDQN, DQN,\
    DoubleDQN, MaxminDQN, DuelingDQN, NoisyDQN, QuantileDQN, Rainbow
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory

from atari_gymnasium_wrapper import AtariGymnasiumWrapper


class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(-1, 3136)))
        q = self._h5(h)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, Network.n_features)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        h = F.relu(self._h1(state.float() / 255.))
        h = F.relu(self._h2(h))
        h = F.relu(self._h3(h))
        h = F.relu(self._h4(h.view(-1, 3136)))

        return h


class DeepCRL:
    def __init__(self, env, screen_width=84, screen_height=84,
                 initial_replay_size=50000, max_replay_size=500000, prioritized=False,
                 optimizer='adam', learning_rate=0.0001, decay=0.95, epsilon=1e-8,
                 algorithm='dqn', n_approximators=1, batch_size=32, history_length=4,
                 target_update_frequency=10000, evaluation_frequency=250000, train_frequency=4,
                 max_steps=50000000, final_exploration_frame=1000000, initial_exploration_rate=1.0,
                 final_exploration_rate=0.1, test_exploration_rate=0.05, test_samples=125000,
                 max_no_op_actions=30, alpha_coeff=0.6, n_atoms=51, v_min=-10, v_max=10,
                 n_quantiles=200, n_steps_return=3, sigma_coeff=0.5,
                 use_cuda=False, save=False, load_path=None, render=False, quiet=False, debug=False):

        # Environment
        self.name = env.spec.id  # Gym ID of the Atari game
        self.screen_width = screen_width  # Width of the game screen
        self.screen_height = screen_height  # Height of the game screen

        self.env = env
        self.states = env.states
        self.actions = env.actions
        self.reward_variable_values = env.reward_variable_values
        self.reward_variable_categories = env.reward_variable_categories
        # Counting the number of relational states
        self.relational_states_count = 1
        for i in env.state_variables_cardinalities:
            self.relational_states_count *= i
        # Creating a dic to count the number of times each action is performed in a given original state
        self.original_action_count = np.zeros((len(self.states), len(self.actions)))
        # Creating a dic to count the number of times each action is performed in a given relational state
        self.relational_action_count = np.zeros((len(self.states), len(self.actions)))

        # Replay Memory
        self.initial_replay_size = initial_replay_size  # Initial size of the replay memory
        self.max_replay_size = max_replay_size  # Max size of the replay memory
        self.prioritized = prioritized  # Whether to use prioritized memory or not

        # Deep Q-Network
        self.optimizer = optimizer  # Name of the optimizer to use
        self.learning_rate = learning_rate  # Learning rate value of the optimizer
        self.decay = decay  # Discount factor for the history coming from the gradient momentum

        # Algorithm
        self.algorithm = algorithm  # Name of the algorithm
        self.n_approximators = n_approximators  # Number of approximators used
        self.batch_size = batch_size  # Batch size for each fit of the network
        self.history_length = history_length  # Number of frames composing a state
        self.target_update_frequency = target_update_frequency  # Frequency of target network update
        self.evaluation_frequency = evaluation_frequency  # Frequency of evaluation
        self.train_frequency = train_frequency  # Frequency of network training
        self.max_steps = max_steps  # Total number of collected samples
        self.final_exploration_frame = final_exploration_frame  # Number of samples for exploration rate decay
        self.initial_exploration_rate = initial_exploration_rate  # Initial exploration rate
        self.final_exploration_rate = final_exploration_rate  # Final exploration rate
        self.test_exploration_rate = test_exploration_rate  # Exploration rate during evaluation
        self.test_samples = test_samples  # Number of collected samples for each evaluation
        self.max_no_op_actions = max_no_op_actions  # Maximum number of no-op actions at the beginning of episodes
        self.alpha_coeff = alpha_coeff  # Prioritization exponent
        self.n_atoms = n_atoms  # Number of atoms for Categorical DQN
        self.v_min = v_min  # Minimum action-value for Categorical DQN
        self.v_max = v_max  # Maximum action-value for Categorical DQN
        self.n_quantiles = n_quantiles  # Number of quantiles for Quantile Regression DQN
        self.n_steps_return = n_steps_return  # Number of steps for n-step return for Rainbow
        self.sigma_coeff = sigma_coeff  # Sigma0 coefficient for noise initialization

        # Util
        self.use_cuda = use_cuda  # Flag specifying whether to use the GPU
        self.save = save  # Flag specifying whether to save the model
        self.load_path = load_path  # Path of the model to be loaded
        self.render = render  # Flag specifying whether to render the game
        self.quiet = quiet  # Flag specifying whether to hide the progress bar
        self.debug = debug  # Flag specifying whether the script runs in debug mode

        # Summary folder
        self.folder_name = './logs/atari_' + algorithm_name + '_' + self.name + \
                      '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        pathlib.Path(self.folder_name).mkdir(parents=True)

        self.optimizer = dict()
        if optimizer == 'adam':
            self.optimizer['class'] = optim.Adam
            self.optimizer['params'] = dict(lr=learning_rate,
                                       eps=epsilon)
        elif optimizer == 'adadelta':
            self.optimizer['class'] = optim.Adadelta
            self.optimizer['params'] = dict(lr=learning_rate,
                                       eps=epsilon)
        elif optimizer == 'rmsprop':
            self.optimizer['class'] = optim.RMSprop
            self.optimizer['params'] = dict(lr=learning_rate,
                                       alpha=decay,
                                       eps=epsilon)
        elif optimizer == 'rmspropcentered':
            self.optimizer['class'] = optim.RMSprop
            self.optimizer['params'] = dict(lr=learning_rate,
                                       alpha=decay,
                                       eps=epsilon,
                                       centered=True)

        # MDP initialization for TaxiAtari
        self.mdp = AtariGymnasiumWrapper(self.env, screen_width, screen_height,
                                    ends_at_life=True, history_length=history_length,
                                    max_no_op_actions=0,max_steps_before_reset=max_steps_before_reset)

        # Policy
        self.epsilon = LinearParameter(value=initial_exploration_rate,
                                  threshold_value=final_exploration_rate,
                                  n=final_exploration_frame)
        self.epsilon_test = Parameter(value=test_exploration_rate)
        self.epsilon_random = Parameter(value=1)
        self.pi = EpsGreedy(epsilon=self.epsilon_random)

        # Approximator
        self.approximator_params = dict(
            network=Network if algorithm not in ['dueldqn', 'cdqn', 'ndqn', 'qdqn', 'rainbow'] else FeatureNetwork,
            input_shape=self.mdp.info.observation_space.shape,
            output_shape=(self.mdp.info.action_space.n,),
            n_actions=self.mdp.info.action_space.n,
            n_features=Network.n_features,
            optimizer=self.optimizer,
            use_cuda=use_cuda
        )
        if self.algorithm not in ['cdqn', 'qdqn', 'rainbow']:
            self.approximator_params['loss'] = F.smooth_l1_loss

        self.approximator = TorchApproximator

        # Memory
        if prioritized:
            self.replay_memory = PrioritizedReplayMemory(
                initial_replay_size, max_replay_size, alpha=alpha_coeff,
                beta=LinearParameter(.4, threshold_value=1,
                                     n=max_steps // train_frequency)
            )
        else:
            self.replay_memory = None

        # Agent
        self.algorithm_params = dict(
            batch_size=batch_size,
            target_update_frequency=target_update_frequency // train_frequency,
            replay_memory=self.replay_memory,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size
        )

        if algorithm == 'dqn':
            alg = DQN
            self.agent = alg(self.mdp.info, self.pi, self.approximator,
                        approximator_params=self.approximator_params,
                        **self.algorithm_params)
        elif algorithm == 'ddqn':
            alg = DoubleDQN
            self.agent = alg(self.mdp.info, self.pi, self.approximator,
                        approximator_params=self.approximator_params,
                        **self.algorithm_params)
        elif algorithm == 'adqn':
            alg = AveragedDQN
            self.agent = alg(self.mdp.info, self.pi, self.approximator,
                        approximator_params=self.approximator_params,
                        n_approximators=n_approximators,
                        **self.algorithm_params)
        elif algorithm == 'mmdqn':
            alg = MaxminDQN
            self.agent = alg(self.mdp.info, self.pi, self.approximator,
                        approximator_params=self.approximator_params,
                        n_approximators=n_approximators,
                        **self.algorithm_params)
        elif algorithm == 'dueldqn':
            alg = DuelingDQN
            self.agent = alg(self.mdp.info, self.pi, approximator_params=self.approximator_params,
                        **self.algorithm_params)
        elif algorithm == 'cdqn':
            alg = CategoricalDQN
            self.agent = alg(self.mdp.info, self.pi, approximator_params=self.approximator_params,
                        n_atoms=n_atoms, v_min=v_min,
                        v_max=v_max, **self.algorithm_params)
        elif algorithm == 'ndqn':
            alg = NoisyDQN
            self.agent = alg(self.mdp.info, self.pi, approximator_params=self.approximator_params,
                        sigma_coeff=sigma_coeff, **self.algorithm_params)
        elif algorithm == 'qdqn':
            alg = QuantileDQN
            self.agent = alg(self.mdp.info, self.pi, approximator_params=self.approximator_params,
                        n_quantiles=n_quantiles, **self.algorithm_params)
        elif algorithm == 'rainbow':
            alg = Rainbow
            beta = LinearParameter(.4, threshold_value=1, n=max_steps // train_frequency)
            self.agent = alg(self.mdp.info, self.pi, approximator_params=self.approximator_params,
                        n_atoms=n_atoms, v_min=v_min,
                        v_max=v_max, n_steps_return=n_steps_return,
                        alpha_coeff=alpha_coeff, beta=beta,
                        sigma_coeff=sigma_coeff, **self.algorithm_params)

        self.logger = Logger(alg.__name__, results_dir=None)
        self.logger.strong_line()
        self.logger.info('Experiment Algorithm: ' + alg.__name__)

        # Algorithm
        self.core = Core(self.agent, self.mdp)

    def crl_learn(self, max_ep, random_initial_states):

        episode_reward = []  # cumulative reward
        steps_per_episode = []  # steps per episodes
        actions_by_model_count_list = []
        good_actions_by_model_count_list = []
        episode_stage = []  # to store the corresponding stage on each episode

        # SHD distances to measure the performance of Causal Discovery
        shd_distances = {}

        # To save RL data for all interest variables
        data_set = {}
        for action_name in self.env.actions:
            data_set[action_name] = {"all_states_i": [], "all_states_j": [], "all_rewards": []}
            shd_distances[action_name] = []  # Just initializing

        # Initialize data_set with synthetic examples for all actions
        # First, find the variable with higher cardinality and add one row with each possible value for that variable.
        # Then fill the table for the other variables using the corresponding value % cardinality(other variable)
        levels_variables = self.env.state_variables_cardinalities
        max_level_value = max(levels_variables)
        variables_count = len(levels_variables)

        max_value = max(max_level_value, self.env.reward_variable_cardinality)

        synthetic_values = []

        for i in range(max_value):
            row = []
            for j in range(variables_count):
                value = i % levels_variables[j]
                row.append(value)
            synthetic_values.append(row)

        for action in data_set.keys():
            for i in range(len(synthetic_values)):
                data_set[action]['all_states_i'].append(synthetic_values[i])
                data_set[action]['all_states_j'].append(synthetic_values[i])
                data_set[action]['all_rewards'].append(i % self.env.reward_variable_cardinality)

        actual_episodes = 0
        maximum_episodes_reached = False
        causal_models = None

        for stage in combination_strategy.stages:

            stage_times = stage.times

            while True and not maximum_episodes_reached:
                steps = stage.steps
                for step in steps:

                    step_length_in_episodes = 0 # This is the default length for CD and Model Init

                    if step == Step.MODEL_INIT:
                        sys.stdout.write("\rEpisode {}. Loading models for CM initialization".format(actual_episodes))
                        sys.stdout.flush()
                        #print("Loading models for CM initialization")
                        episode_stage.append((Step.MODEL_INIT.value, actual_episodes, actual_episodes))
                        causal_models, structural_distances = self.causal_discovery_using_rl_data(
                            model_init_path)

                    elif step == Step.CD:
                        sys.stdout.write("\rEpisode {}. Learning CMs using RL data".format(actual_episodes))
                        #print("Learning CMs using RL data")
                        episode_stage.append((Step.CD.value, actual_episodes, actual_episodes))
                        causal_models, structural_distances = self.causal_discovery_using_rl_data(directory_path)
                        # Saving the structural distance for latter plotting
                        for model in causal_models:
                            shd_distances[model].append(structural_distances[model])

                    else:
                        # Checking if we can perform T steps of we need to do less
                        if actual_episodes + T <= max_ep:
                            step_length_in_episodes = T

                        elif actual_episodes + T > max_ep:
                            step_length_in_episodes = max_ep - actual_episodes
                            print(
                                "It is not possible to execute the next step of the current stage for T episodes. Instead we are running until the max_ep")

                        if step == Step.RL:
                            sys.stdout.write("\rEpisode {}. Learning task policy using classical RL for the next {} episodes".format(actual_episodes, step_length_in_episodes))
                            #print("Learning task policy using classical RL for the next {} episodes".format(
                            #    step_length_in_episodes))
                            episode_stage.append(
                                (Step.RL.value, actual_episodes, actual_episodes + step_length_in_episodes))

                            c_qlearning_r, c_qlearning_steps, c_record, c_actions_by_model_count, c_good_actions_by_model_count, epi_sta = self.learn(
                                step_length_in_episodes,
                                actual_episodes, step, random_initial_states)

                            # Add the RL observations in record to the cumulative data_set
                            for action in self.actions:
                                data_set[action]['all_states_i'].extend(c_record[action]['all_states_i'])
                                data_set[action]['all_states_j'].extend(c_record[action]['all_states_j'])
                                data_set[action]['all_rewards'].extend(c_record[action]['all_rewards'])

                            # Save the dataset to files to perform Causal Discovery
                            directory_path = self.save_data_to_file(data_set, actual_episodes + step_length_in_episodes)

                            episode_reward.extend(c_qlearning_r)
                            steps_per_episode.extend(c_qlearning_steps)

                        elif step == Step.RL_USING_CD:
                            sys.stdout.write(
                                "\rEpisode {}. Using the causal models for the next {} episodes".format(
                                    actual_episodes, step_length_in_episodes))
                            #print("Using the causal models for the next {} episodes".format(step_length_in_episodes))
                            episode_stage.append(
                                (Step.RL_USING_CD.value, actual_episodes, actual_episodes + step_length_in_episodes))

                            c_qlearning_r, c_qlearning_steps, c_record, c_actions_by_model_count, c_good_actions_by_model_count, epi_sta = self.learn(
                                step_length_in_episodes, actual_episodes, step, random_initial_states, causal_models)
                            actions_by_model_count_list.append(c_actions_by_model_count)
                            good_actions_by_model_count_list.append(c_good_actions_by_model_count)

                            if use_crl_data:
                                # Add the RLCM observations in record to the cumulative data_set
                                for action in self.actions:
                                    data_set[action]['all_states_i'].extend(c_record[action]['all_states_i'])
                                    data_set[action]['all_states_j'].extend(c_record[action]['all_states_j'])
                                    data_set[action]['all_rewards'].extend(c_record[action]['all_rewards'])

                                # Save the dataset to files to perform Causal Discovery
                                directory_path = self.save_data_to_file(data_set, actual_episodes + step_length_in_episodes)

                            episode_reward.extend(c_qlearning_r)
                            steps_per_episode.extend(c_qlearning_steps)

                        elif step == Step.RL_FOR_CD:
                            sys.stdout.write(
                                "\rEpisode {}. Learning task policy using actions focused on CD for the next {} episodes".format(
                                    actual_episodes, step_length_in_episodes))
                            #print("Learning task policy using actions focused on CD for the next {} episodes".format(step_length_in_episodes))
                            episode_stage.append(
                                (Step.RL_FOR_CD.value, actual_episodes, actual_episodes + step_length_in_episodes))

                            c_qlearning_r, c_qlearning_steps, c_record, c_actions_by_model_count, c_good_actions_by_model_count, epi_sta = self.learn(
                                step_length_in_episodes, actual_episodes, step, random_initial_states)

                            # Add the RL observations in record to the cumulative data_set
                            for action in self.actions:
                                data_set[action]['all_states_i'].extend(c_record[action]['all_states_i'])
                                data_set[action]['all_states_j'].extend(c_record[action]['all_states_j'])
                                data_set[action]['all_rewards'].extend(c_record[action]['all_rewards'])

                            # Save the dataset to files to perform Causal Discovery
                            directory_path = self.save_data_to_file(data_set, actual_episodes + step_length_in_episodes)

                            episode_reward.extend(c_qlearning_r)
                            steps_per_episode.extend(c_qlearning_steps)


                    # update the variables for next cycle
                    actual_episodes += step_length_in_episodes

                    if actual_episodes >= max_ep:
                        maximum_episodes_reached = True
                        sys.stdout.write("\rEpisode {}. Stages completed".format(actual_episodes))
                        sys.stdout.flush()
                        # print("Stages completed\n")
                        break

                if stage_times == Times.ONE or maximum_episodes_reached:
                    break

        return np.array(episode_reward), np.array(
            steps_per_episode), c_record, actions_by_model_count_list, good_actions_by_model_count_list, shd_distances, episode_stage

        # Loop forever:
        #     for n_steps do
        #         observe_the_enviroment()
        #    for n_steps do
        #         reinforcement_learning()
        #    causal_model = causal_discovery_usign_data()
        #     for n_stpes do // Planeacion y mejora y validar el modelo
        #         mejorar_el_modelo() //Por ahora sin esto seleccion_de_acciones(mejor_recompenza, mas_aporte_al_descubrimiento)
        #         rl_using_causal_model() // mejora guiada por intervenciones

    def learn(self, total_steps, initial_epsilon_index, step_name, causal_models=None):
        """ Perform Deep-RL learning, return episode_reward, episode_steps, and record of rl_data """

        # Set the core to perform the corresponding CARL stage0
        self.core.step_name = step_name

        episode_reward = []  # cumulative reward
        steps_per_episode = []  # steps per episodes
        episode_stage = []  # to store the corresponding stage on each episode
        episode_stage.append((step_name, initial_epsilon_index, initial_epsilon_index + total_steps))

        record = {}

        cm_fited = {}
        # Load the causal models once, to use later
        if causal_models is not None:
            for action_name in self.env.actions:
                cm_fited[action_name] = load_model_function_r(causal_models[action_name])

        for action_name in self.env.actions:
            record[action_name] = {"all_states_i": [], "all_states_j": [], "all_rewards": []}

        # To count the number of times the model suggest an action for the agent
        actions_by_model_count = 0
        good_actions_by_model_count = 0

        # To store the suggested action for each state when causal models are available and avoid re-calculate them
        # This variable is updated at the beginning of every T episodes because the models can be updated
        suggested_action = {}

        # if episode_state_initialization == EpisodeStateInitialization.RANDOM:
        #     seed = None #TODO ver como hago esto

        action_count = self.original_action_count
        if action_count_strategy == ActionCountStrategy.Relational:
            action_count = self.relational_action_count

        scores = list()

        # RUN

        # Fill replay memory with a number of interactions depending on the step_name
        # TODO Do this only if the memory is empty
        util.print_epoch(0, self.logger)
        self.core.learn(n_steps=self.initial_replay_size, n_steps_per_fit=self.initial_replay_size, quiet=quiet)

        if self.save:
            self.agent.save(self.folder_name + '/agent_0.msh')

        # Evaluate initial policy
        self.pi.set_epsilon(self.epsilon_test)
        self.mdp.set_episode_end(False)
        dataset = self.core.evaluate(n_steps=test_samples, render=self.render, quiet=self.quiet)
        scores.append(util.get_stats(dataset, self.logger))

        np.save(self.folder_name + '/scores.npy', scores)

        # For each epoch
        for n_epoch in range(1, max_steps // self.evaluation_frequency + 1):
            util.print_epoch(n_epoch, self.logger)
            self.logger.info('- Learning:')
            # learning step
            self.pi.set_epsilon(self.epsilon)
            self.mdp.set_episode_end(True)
            self.core.learn(n_steps=self.evaluation_frequency,
                       n_steps_per_fit=self.train_frequency, quiet=self.quiet)

            if self.save:
                self.agent.save(self.folder_name + '/agent_' + str(n_epoch) + '.msh')

            self.logger.info('- Evaluation:')
            # evaluation step
            self.pi.set_epsilon(self.epsilon_test)
            self.mdp.set_episode_end(False)
            dataset = self.core.evaluate(n_steps=self.test_samples, render=self.render,
                                    quiet=self.quiet)
            scores.append(util.get_stats(dataset, self.logger))

            np.save(self.folder_name + '/scores.npy', scores)

        # For each epoch
        for episode in range(total_episodes):

            if not shared_initial_states:
                random_initial_states = None

            if random_initial_states is not None:
                start_state, _ = self.env.reset(options={'state_index': random_initial_states[initial_epsilon_index + episode], 'state_type': "original"})
            elif episode_state_initialization == EpisodeStateInitialization.SAME:
                start_state, _ = self.env.reset(options={'state_index': episode_state_initialization.value, 'state_type': "original"})
            elif episode_state_initialization == EpisodeStateInitialization.RANDOM:
                start_state, _ = self.env.reset(seed = seed)
            elif episode_state_initialization == EpisodeStateInitialization.EPISODE_NUMBER:
                start_state, _ = self.env.reset(options={'state_index': initial_epsilon_index + episode, 'state_type': "original"})
            elif episode_state_initialization == EpisodeStateInitialization.RELATIONAL_EPISODE_NUMBER:
                start_state, _ = self.env.reset(options={'state_index': initial_epsilon_index + episode, 'state_type': "original"})

            # self.env.render()
            self.states = self.env.states  # Updating the state list in relational representation

            steps = 0
            cumulative_reward = 0
            terminated = False

            while steps < self.max_steps and not terminated:

                # s = self.states.index(current_state)  # Parsing the state to a given index
                s = self.env.s

                state_index = s # This variable store the corresponding original state or relational state

                # getting the corresponding rlstate number
                if action_count_strategy == ActionCountStrategy.Relational:
                    rl_index = 0
                    for i in range(len(self.states[s])):
                        rl_index += self.states[s][i]

                        if i+1 < len(self.states[s]):
                            rl_index *= self.env.state_variables_cardinalities[i+1]
                    state_index = rl_index

                # self.env.render(mode = "human",
                #     info={'episode_number': str(episode), 'step_number': str(steps), "reward": str(cumulative_reward)})

                # Uncomment to export the given frame to a file
                # plt.imshow(frame)
                # plt.axis('off')
                # plt.savefig('frame.png', bbox_inches='tight', pad_inches=0)

                a = None
                action_indexes = []
                # action_rewards = []
                do_rl = False

                if step_name == Step.RL_USING_CD:

                    # Check if the model already gives an action on the given state to not infer it again
                    if state_index in suggested_action:
                        action_indexes, reward_indexes = suggested_action[state_index]

                    elif model_use_strategy == ModelUseStrategy.IMMEDIATE_POSITIVE:

                        action_indexes, reward_indexes = self.causal_based_action_selection_1(self.states[self.env.s],
                                                                                              cm_fited)

                    elif model_use_strategy == ModelUseStrategy.POSITIVE_OR_NOT_NEGATIVE:

                        action_indexes, reward_indexes = self.causal_based_action_selection_2(self.states[self.env.s],
                                                                                              cm_fited)

                    elif model_use_strategy == ModelUseStrategy.DATA_AUGMENTATION:
                        pass  # TODO Implement this part
                        # action_rewards = self.causal_based_action_selection_3(self.states[self.env.s], cm_fited)
                        # # Use the expected rewards and expected transition to update Q without actually perform any step
                        # for action_index in range(0, len(action_rewards)):
                        #     expected_reward = self.env.reward_variable_values[action_rewards[action_index]]
                        #     expected_s_prime = 0  # TODO ver como estimar bien aqui
                        #     self.q[s, action_index] += self.alpha * (
                        #             expected_reward + self.gamma * np.max(self.q[s_prime]) - self.q[s, action_index])

                    suggested_action[state_index] = action_indexes, reward_indexes

                    if action_indexes:
                        if crl_action_selection_strategy == ActionSelectionStrategy.MODEL_BASED_EPSILON_GREEDY:

                            # Then, use the same epsilon-greedy policy to select among the filtered actions_indexes
                            if np.random.uniform() < self.epsilon_values[initial_epsilon_index + episode]:  # Explore
                                a = np.random.choice(action_indexes)
                                # a = self.actions.index(np.random.choice(self.actions))
                                # a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],action_indexes))
                            else:  # Exploit selecting the best action according Q
                                a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                                #a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0], action_indexes))

                        elif crl_action_selection_strategy == ActionSelectionStrategy.NEW_IDEA:

                            # If we know the model is good, we can set the Q-values for non-filtered actions to negative
                            first = np.arange(len(self.actions))
                            second = np.array(action_indexes)
                            diff = np.setdiff1d(first, second)
                            self.q[s, diff] = -1000.0

                            # Then, use the same epsilon-greedy policy to select among the filtered actions_indexes
                            if np.random.uniform() < self.epsilon_values[initial_epsilon_index + episode]:  # Explore
                                a = np.random.choice(action_indexes)
                                # a = self.actions.index(np.random.choice(self.actions))
                                # a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],action_indexes))
                            else:  # Exploit selecting the best action according Q
                                # a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                                a = np.random.choice(
                                    np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],
                                                   action_indexes))

                        elif crl_action_selection_strategy == ActionSelectionStrategy.RANDOM_MODEL_BASED:
                            # This is another option, always select a random action among the suggested by the model
                            a = np.random.choice(action_indexes)

                        actions_by_model_count = actions_by_model_count + 1
                        # First clone the environment at this point to preserve the current state
                        # clone_env = copy.deepcopy(self.env)
                        # To test if the suggested action was good, despite the environment stochastic, perform an step on the cloned deterministic environment
                        # next_state_d, r_d, done_d, info_d = self.step(current_state, self.actions[a], "deterministic")
                        # clone_env.env_type = "deterministic"
                        # observation_d, reward_d, terminated_d, truncated_d, info_d = clone_env.step(a)
                        # An action is considered good if the expected reward is equal to the observed reward
                        # TODO Implement this part good after implementing the action decision
                        # if reward == self.env.reward_variable_values[reward_indexes[action_indexes.index(a)]]:
                        good_actions_by_model_count = good_actions_by_model_count + 1

                    else:  # The model does not suggest any action, then do RL
                        do_rl = True

                elif step_name == Step.RL_FOR_CD:

                    if model_discovery_strategy == ModelDiscoveryStrategy.LESS_SELECTED_ACTION_EPSILON_GREEDY:
                        # Then, use the same epsilon-greedy policy
                        if np.random.uniform() < self.epsilon_values[initial_epsilon_index + episode]:
                            # Explore but considering the best action for CD

                            candidates_actions = np.where(action_count[state_index] < min_frequency)

                            if candidates_actions[0].size > 0:
                                a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])
                                # a = np.random.choice(candidates_actions[0])
                            else:
                                a = self.actions.index(np.random.choice(self.actions))

                            # a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])
                            # candidates_actions = np.where(self.action_count[rl_index] < min_frequency)
                            #
                            # if candidates_actions[0].size > 0:
                            #     a = np.random.choice(candidates_actions[0])
                            #
                            # else:
                            #     a = self.actions.index(np.random.choice(self.actions))

                        else:  # Exploit selecting the best action according Q
                            a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                            # a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],action_indexes))

                    elif model_discovery_strategy == ModelDiscoveryStrategy.NEW_IDEA:
                        # Aqui una idea puede ser hacer siempre acciones que considero importantes para
                        # CD independientemente de epsilon y de lo que diga Q.
                        a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])

                    elif model_discovery_strategy == ModelDiscoveryStrategy.RANDOM:
                        a = self.actions.index(np.random.choice(self.actions))

                    elif model_discovery_strategy == ModelDiscoveryStrategy.EPSILON_GREEDY:
                        if np.random.uniform() < self.epsilon_values[initial_epsilon_index + episode]:  # Explore
                            a = self.actions.index(np.random.choice(self.actions))
                        else:  # Exploit
                            a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])

                    else: # Model discovery strategy is to select the same action always
                        a = model_discovery_strategy.value

                if step_name == Step.RL or do_rl:

                    do_rl = False

                    if rl_action_selection_strategy == ActionSelectionStrategy.EPSILON_GREEDY:
                        if np.random.uniform() < self.epsilon_values[initial_epsilon_index + episode]:
                            # Explore
                            a = self.actions.index(np.random.choice(self.actions))

                        else:  # Exploit
                            a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])

                # Update the action_count variable
                #if step_name == Step.RL_FOR_CD:
                action_count[state_index, a] += 1

                # Take action, observe outcome
                observation, reward, terminated, truncated, info = self.env.step(a)

                # s_prime = self.states.index(next_state)
                s_prime = observation

                # Save the info for causal discovery
                record[self.actions[a]]["all_states_i"].append(self.states[s])
                record[self.actions[a]]["all_states_j"].append(self.states[s_prime])
                record[self.actions[a]]["all_rewards"].append(
                    self.reward_variable_categories[self.reward_variable_values.index(reward)])

                steps = steps + 1
                cumulative_reward = cumulative_reward + reward

                # Q-Learning
                update = self.alpha * (reward + self.gamma * np.max(self.q[s_prime]) - self.q[s, a])
                self.q[s, a] += update

                # Set state for next loop. NOT NECESSARY ANY MORE

            # Add the corresponding evaluation metric for later plotting
            if evaluation_metric == EvaluationMetric.EPISODE_REWARD:
                episode_reward.append(cumulative_reward)
                steps_per_episode.append(steps)
            elif evaluation_metric == EvaluationMetric.CURRENT_Q:
                # Calculate the reward obtained using the actual values of Q instead episode reward
                actual_reward, actual_steps = self.env.do_task(start_state, self.max_steps, self.q)
                episode_reward.append(actual_reward)
                steps_per_episode.append(actual_steps)

            if print_info:

                if steps != self.max_steps:
                    sys.stdout.write(
                        "\rEpisode {} Step {}. The agent completed the task in {} steps".format(
                            initial_epsilon_index + episode, step_name.value, steps))
                else:
                    sys.stdout.write(
                        "\rEpisode {} Step {}. Maximum episode steps reached".format(initial_epsilon_index + episode,
                                                                                     step_name.value))

        return np.array(episode_reward), np.array(
            steps_per_episode), record, actions_by_model_count, good_actions_by_model_count, episode_stage

    def causal_based_action_selection_1(self, current_state, cm_fitted):

        probs = []

        if cm_fitted is not None:

            for action in self.actions:
                # Load the stored cm_fitted object for the given action
                dbn_fit = cm_fitted[action]
                # Calculate the probability query of reward given the current state
                prob = dbn_inference_function_r(self.env.spec.name, environment_type, ro.IntVector(current_state), dbn_fit)
                probs.append(list(prob))

            action_indexes = []  # To store the indexes of good actions
            reward_indexes = []  # To store the indexes of good rewards

            # Check for good actions at probs[0]
            for i in range(len(probs)):
                if probs[i][0] >= threshold:
                    action_indexes.append(i)
                    reward_indexes.append(0)

            return action_indexes, reward_indexes

        return None

    def causal_based_action_selection_2(self, current_state, cm_fitted):

        # To store the probability of reward given the current state and action
        probs = []

        if cm_fitted is not None:

            for action in self.actions:
                # Load the stored cm_fitted object for the given action
                dbn_fit = cm_fitted[action]
                # Calculate the probability query of reward given the current state
                prob = dbn_inference_function_r(self.env.spec.name, environment_type, ro.IntVector(current_state), dbn_fit)
                probs.append(list(prob))

            # Check all elements in probs table to find the values of high reward higher than threshold and to filter
            # the actions with high probability to give very-low reward

            # This variable will contain either the index of a good action or the indexes of possible actions.
            action_indexes = list(range(0, len(probs)))  # Initially all the actions indexes are possible
            # TODO Define this later
            # reward_indexes = list(range(0, len(probs)))  # Initially all the reward indexes are possible

            # Check for good actions at probs[0] and bad actions at probs[-1]
            for i in range(len(probs)):
                if probs[i][0] >= threshold:
                    return [i], [0]  # If we found a good action, return a list containing the index
                elif probs[i][-1] >= threshold:
                    action_indexes.remove(i)

            return action_indexes, None

        return None

    def causal_based_action_selection_3(self, current_state, cm_fitted):

        # To store the estimated reward given the current state and action
        probs = []  # TODO, Calcular tambien la probabilidad de transicion y devolver ambas

        if cm_fitted is not None:

            for action in self.actions:
                # Load the stored cm_fited object for the given action
                dbn_fit = cm_fitted[action]
                # Calculate the probability query of reward give the current state
                prob = dbn_inference_function_r(self.env.spec.name, environment_type, ro.IntVector(current_state), dbn_fit)
                probs.append(np.argmax(list(prob)))

            return probs

        return None

    def causal_discovery_using_rl_data(self, directory_path):

        causal_models = {}
        structural_distances = {}

        # Save data to files before to perform Causal Discovery
        for action in self.actions:
            # Invoking the R function and getting the result
            # The function read the data from file, estimate de model and save it
            # to another file using bnlearn write.net function in HUGIN flat network format,
            # then calculate the SHD respect the ground truth and finally plot and save the comparison
            results = cd_function_r(self.env.spec.name, environment_type, directory_path, action.lower())
            causal_models[action] = results[0][0]
            structural_distances[action] = results[1][0]

        return causal_models, structural_distances

    def save_data_to_file(self, data_set, actual_episodes):
        for action in self.actions:
            all_states_i = data_set[action]["all_states_i"]
            all_states_j = data_set[action]["all_states_j"]
            all_rewards = data_set[action]["all_rewards"]

            folder = results_folder + "/" + experiment_sub_folder_name + "/" + causal_discovery_data_folder + "/" + alg_name

            directory_path = util.save_rl_data(folder, action, all_states_i, all_states_j,
                                               all_rewards, env, epsilon_values, actual_episodes)
        return directory_path


if __name__ == '__main__':

    # Single test
    experiments_to_run = config.exp_deep_rl_1

    # Multiple test
    # experiments_to_run = config.exp_coffee_1 + config.exp_coffee_2 + config.exp_coffee_3 + config.exp_coffee_4 + config.exp_taxi_small_1 + config.exp_taxi_small_2 + config.exp_taxi_small_3 + config.exp_taxi_small_4 + config.exp_taxi_big_1

    for experiment_conf in experiments_to_run:

        # Environment Parameters
        environment_name = experiment_conf.env_name.value
        environment_type = experiment_conf.env_type.value

        reward_type = "original" if environment_type == EnvironmentType.DETERMINISTIC.value else "new"

        # Environment Initialization
        env = gym.make(environment_name, render_mode = "rgb_array", env_type = environment_type, reward_type = reward_type, render_fps=64)

        # Params for the experiment output related folder and names
        results_folder = "Deep RL experiments"
        # Sub folders
        rl_result_folder = "rl_results"
        causal_discovery_data_folder = "cd_data_and_results"

        # A folder to store the Ground Truth models
        ground_truth_models_folder = "ground_truth_models/" + environment_name
        # Calling the R function to plot and save the Ground Truth Models
        plot_gt_funtion_r(env.spec.name, environment_type, ground_truth_models_folder)

        # Initializing experiment related params
        experiment_name = experiment_conf.exp_name
        evaluation_metric = experiment_conf.evaluation_metric
        # TODO actually max_episodes is the max_total_steps
        max_total_steps = experiment_conf.max_episodes
        # TODO actually max_steps is the maximun steps before restart the environment.
        max_steps_before_reset = experiment_conf.max_steps
        action_count_strategy = experiment_conf.action_count_strategy
        shared_initial_states = experiment_conf.shared_initial_states
        trials = experiment_conf.trials
        smooth = experiment_conf.smooth
        number_of_algorithms = len(experiment_conf.alg_confs)

        print_info = True
        # get the start time
        st = time.time()

        experiment_folder_name = util.get_experiment_folder_name(experiment_name, env.spec.name, environment_type, max_total_steps, max_steps_before_reset, action_count_strategy,shared_initial_states, trials)
        print("Starting Experiment: " + experiment_folder_name)

        # Variables to store the results on RL policy learning (reward and steps) among trials for each algorithm
        algorithm_names = [""] * number_of_algorithms
        algorithm_r = np.zeros((number_of_algorithms, trials, max_total_steps))
        algorithm_steps = np.zeros((number_of_algorithms, trials, max_total_steps))

        # Variable to measure CD results of each algorithm performing CD among trials
        causal_qlearning_shd_distances = []  # To store the shd_distances dict for each trial
        # To measure action selection using CM
        causal_qlearning_actions_by_model_count = []
        causal_qlearning_good_actions_by_model_count = []

        for t in range(trials):
            print("\nStarting Trial: " + str(t + 1))
            experiment_sub_folder_name = experiment_folder_name + "/trial " + str(t + 1)

            alg_doing_cd_name = []  # to store the name of the algorithms doing CD
            alg_shd_distances = []
            alg_episode_stages = []

            # Creating the random episode init state vector to be shared among all algorithms on each trial
            #random_initial_states = env.random_initial_states(max_episodes)

            for a in range(number_of_algorithms):

                algorithm = experiment_conf.alg_confs[a]

                alg_name = algorithm.alg_name
                algorithm_names[a] = alg_name
                combination_strategy = algorithm.combination_strategy

                # Params for Deep-Rl algorithm part
                screen_width = algorithm.screen_width  # Width of the game screen
                screen_height = algorithm.screen_height  # Height of the game screen
                # Replay Memory
                initial_replay_size = algorithm.initial_replay_size  # Initial size of the replay memory
                max_replay_size = algorithm.max_replay_size  # Max size of the replay memory
                prioritized = algorithm.prioritized  # Whether to use prioritized memory or not
                # Deep Q-Network
                optimizer = algorithm.optimizer  # Name of the optimizer to use
                learning_rate = algorithm.learning_rate  # Learning rate value of the optimizer
                decay = algorithm.decay  # Discount factor for the history coming from the gradient momentum
                epsilon = algorithm.epsilon  # Epsilon term used in optimizer
                # Algorithm
                algorithm_name = algorithm.algorithm  # Name of the algorithm
                n_approximators = algorithm.n_approximators  # Number of approximators used
                batch_size = algorithm.batch_size  # Batch size for each fit of the network
                history_length = algorithm.history_length  # Number of frames composing a state
                target_update_frequency = algorithm.target_update_frequency  # Frequency of target network update
                evaluation_frequency = algorithm.evaluation_frequency  # Frequency of evaluation
                train_frequency = algorithm.train_frequency  # Frequency of network training
                max_steps = algorithm.max_steps  # Total number of collected samples
                final_exploration_frame = algorithm.final_exploration_frame  # Number of samples for exploration rate decay
                initial_exploration_rate = algorithm.initial_exploration_rate  # Initial exploration rate
                final_exploration_rate = algorithm.final_exploration_rate  # Final exploration rate
                test_exploration_rate = algorithm.test_exploration_rate  # Exploration rate during evaluation
                test_samples = algorithm.test_samples  # Number of collected samples for each evaluation
                max_no_op_actions = algorithm.max_no_op_actions  # Maximum number of no-op actions at the beginning of episodes
                alpha_coeff = algorithm.alpha_coeff  # Prioritization exponent
                n_atoms = algorithm.n_atoms  # Number of atoms for Categorical DQN
                v_min = algorithm.v_min  # Minimum action-value for Categorical DQN
                v_max = algorithm.v_max  # Maximum action-value for Categorical DQN
                n_quantiles = algorithm.n_quantiles  # Number of quantiles for Quantile Regression DQN
                n_steps_return = algorithm.n_steps_return  # Number of steps for n-step return for Rainbow
                sigma_coeff = algorithm.sigma_coeff  # Sigma0 coefficient for noise initialization
                # Util
                use_cuda = algorithm.use_cuda  # Flag specifying whether to use the GPU
                save = algorithm.save  # Flag specifying whether to save the model
                load_path = algorithm.load_path  # Path of the model to be loaded
                render = algorithm.render  # Flag specifying whether to render the game
                quiet = algorithm.quiet  # Flag specifying whether to hide the progress bar
                debug = algorithm.debug  # Flag specifying whether the script runs in debug mode

                # Parameters for Causal-RL
                if isinstance(algorithm, DeepCRLConf):
                    # Episodes to change between RL for CD and RL using CD
                    T = algorithm.T
                    # Causal Discovery Threshold
                    threshold = algorithm.th
                    min_frequency = algorithm.min_frequency
                    model_use_strategy = algorithm.model_use_strategy
                    model_discovery_strategy = algorithm.model_discovery_strategy
                    crl_action_selection_strategy = algorithm.crl_action_selection_strategy
                    use_crl_data = algorithm.use_crl_data
                    model_init_path = algorithm.model_init_path

                # # Precalculating the epsilon values for all episodes
                # epsilon_values = []
                # epsilon_decay_rate = (epsilon_start - epsilon_end) / max_episodes
                #
                # for epi in range(max_episodes):
                #     epsilon_values.append(epsilon_start - epi * epsilon_decay_rate)
                #
                #     # if epsilon_strategy == EpsilonStrategy.DECAYED:
                #     #     # epsilon decreases exponentially --> our agent will explore less and less
                #     #     epsilon_values.append(epsilon * np.exp(-decay_rate * epi))
                #     # elif epsilon_strategy == EpsilonStrategy.FIXED:
                #     #     epsilon_values.append(epsilon)

                # Reset the environment before to start, this can be always original because we are goin to restart again later
                env.reset(options={'state_index': 0, 'state_type': "original"})

                print("\nStarting algorithm {}. {}".format(a+1, alg_name))

                # Check if algorithm is CRL first because CRL extend RL
                if isinstance(algorithm, DeepCRLConf):
                    # Do Deep Reinforcement Learning with CARL
                    agent = DeepCRL(env,screen_width, screen_height, initial_replay_size, max_replay_size, prioritized, optimizer,
                                        learning_rate, decay, epsilon, algorithm_name, n_approximators, batch_size, history_length,
                                        target_update_frequency, evaluation_frequency, train_frequency, max_steps, final_exploration_frame,
                                        initial_exploration_rate, final_exploration_rate, test_exploration_rate, test_samples, max_no_op_actions,
                                        alpha_coeff, n_atoms, v_min, v_max, n_quantiles, n_steps_return, sigma_coeff, use_cuda, save, load_path,
                                        render, quiet, debug)

                    algorithm_r[a][t], algorithm_steps[a][
                        t], record, actions_by_model_count, good_actions_by_model_count, shd_distances, episode_stage = agent.crl_learn(
                        max_total_steps)
                    alg_shd_distances.append(shd_distances)

                    for stage in episode_stage:
                        if stage[0] == Step.CD.value:
                            alg_doing_cd_name.append(alg_name)
                            break;

                    alg_episode_stages.append(episode_stage)

                else: #isinstance(algorithm, DeepRLConf):
                    # Do traditional Deep Reinforcement Learning without using any model
                    agent = DeepCRL(env,screen_width, screen_height, initial_replay_size, max_replay_size, prioritized,
                                    optimizer, learning_rate, decay, epsilon, algorithm_name, n_approximators, batch_size,
                                    history_length, target_update_frequency, evaluation_frequency, train_frequency, max_steps,
                                    final_exploration_frame, initial_exploration_rate, final_exploration_rate, test_exploration_rate,
                                    test_samples, max_no_op_actions, alpha_coeff, n_atoms, v_min, v_max, n_quantiles, n_steps_return, sigma_coeff,
                                    use_cuda, save, load_path, render, quiet, debug)
                    algorithm_r[a][t], algorithm_steps[a][
                        t], record, actions_by_model_count, good_actions_by_model_count, episode_stage = agent.learn(
                        max_total_steps, initial_epsilon_index=0, step_name=Step.RL)

            print()
            # Plot and save the RL results for the given trial for all algorithms. The episode stage variable only contains
            # the stages of the last algorithm to run
            util.plot_rl_results(algorithm_names, algorithm_r[:, t], algorithm_steps[:, t], None, None,
                                 results_folder + "/" + experiment_sub_folder_name + "/" + rl_result_folder, epsilon_start,
                                 evaluation_metric, epsilon_values, episode_stage, smooth)

            if len(alg_doing_cd_name) > 0:
                # Plot and save the CD results for the given trial for all algorithms performing CD
                util.plot_cd_results(alg_doing_cd_name, alg_shd_distances,
                                     results_folder + "/" + experiment_sub_folder_name + "/" + causal_discovery_data_folder,
                                     alg_episode_stages, epsilon_values)

                # Adding the alg_shd_distances of the given trial to the global variable
                causal_qlearning_shd_distances.append(alg_shd_distances)

        # # # Average RL results across trials
        algorithm_r_mean = np.zeros((number_of_algorithms, max_episodes))
        algorithm_steps_mean = np.zeros((number_of_algorithms, max_episodes))

        algorithm_r_std = np.zeros((number_of_algorithms, max_episodes))
        algorithm_steps_std = np.zeros((number_of_algorithms, max_episodes))

        for a in range(number_of_algorithms):
            algorithm_r_mean[a] = np.mean(algorithm_r[a], axis=0)
            algorithm_steps_mean[a] = np.mean(algorithm_steps[a], axis=0)

            algorithm_r_std[a] = np.std(algorithm_r[a], axis=0)
            algorithm_steps_std[a] = np.std(algorithm_steps[a], axis=0)

        util.plot_rl_results(algorithm_names, algorithm_r_mean, algorithm_steps_mean, algorithm_r_std, algorithm_steps_std,
                             results_folder + "/" + experiment_folder_name + "/average/" + rl_result_folder, epsilon_start,
                             evaluation_metric, epsilon_values, None,
                             0)  # At this point the data is already smoothed so we dont need to smooth again

        if len(causal_qlearning_shd_distances) > 0:

            #cd_times = len(causal_qlearning_shd_distances[0][1][list(shd_distances.keys())[0]])

            # Average CD results over each dict entry in causal_qlearning_shd_distances
            alg_average_distances = []  # To calculate the average distances of each algorithm doing CD {}
            alg_total_shd_mean = []
            alg_total_shd_std = []

            alg_cd_times = [] # To store the times each algorithm perform CD

            for i in range(len(causal_qlearning_shd_distances[0])):
                alg_cd_times.append(len(causal_qlearning_shd_distances[0][i][list(shd_distances.keys())[0]]))
                alg_total_shd_mean.append(np.zeros(alg_cd_times[i]))
                alg_total_shd_std.append(np.zeros(alg_cd_times[i]))

            for index in range(len(alg_doing_cd_name)):
                average_distances = {}
                # First we need to store in an array all the shd_distances among trials for the given algorithm
                shd_distances = []

                # Total_shd_distances
                total_shd_distances = []

                for t in range(trials):
                    temp = causal_qlearning_shd_distances[t][index]
                    shd_distances.append(temp)
                    total_shd_trial_distance = []
                    for i in range(alg_cd_times[index]):
                        total_shd = 0
                        for action_name in temp:
                            total_shd += temp[action_name][i]
                        total_shd_trial_distance.append(total_shd)
                    total_shd_distances.append(total_shd_trial_distance)

                alg_total_shd_mean[index] = np.mean(total_shd_distances, axis=0)
                alg_total_shd_std[index] = np.std(total_shd_distances, axis=0)

                for trial_distances in shd_distances:
                    for action_name in trial_distances:
                        if action_name in average_distances:
                            average_distances[action_name] = np.add(average_distances[action_name],
                                                                    np.array(trial_distances[action_name]))

                        else:
                            average_distances[action_name] = np.array(trial_distances[action_name])

                for action_name in average_distances:
                    list_values = average_distances[action_name].tolist()
                    average_distances[action_name] = [x / trials for x in list_values]

                alg_average_distances.append(average_distances)

            # Plot and save the CD average results for each algorithm performing CD
            util.plot_cd_results(alg_doing_cd_name, alg_average_distances,
                                 results_folder + "/" + experiment_folder_name + "/average/cd_results/",
                                 alg_episode_stages, epsilon_values)

            # Esto lo voy a comentar porque ya lo tengo calculado arriba
            # Calculating the shd total distance for each time T. The sum of the shd for each action at time T
            # alg_total_shd_distances = []
            #
            # for index in range(len(alg_doing_cd_name)):
            #     actions_shd_distances = alg_average_distances[index]
            #     action_names = env.actions
            #     alg_total = []
            #     for time_step in range(len(actions_shd_distances[action_names[0]])):
            #         shd_sum = 0
            #         for action in action_names:
            #             shd_sum += actions_shd_distances[action][time_step]
            #         alg_total.append(shd_sum)
            #     alg_total_shd_distances.append(alg_total)

            # Plot and save the CD average results for each algorithm performing CD
            util.plot_total_cd_results(alg_doing_cd_name, alg_total_shd_mean, alg_total_shd_std, results_folder + "/" + experiment_folder_name + "/average/cd_results/", alg_episode_stages, epsilon_values)

        # get the execution time estimation
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
