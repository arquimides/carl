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
    EnvironmentType, CARLDQNConf, DQNConf

# Configuring the rpy2 stuff for R communication
r = ro.r  # Creating the R instance
r['source']('causal_discovery.R')  # Loading and sourcing R file.
# Import R functions for later use
cd_function_r = ro.globalenv['causal_discovery_using_rl_data']
load_model_function_r = ro.globalenv['load_model']
dbn_inference_function_r = ro.globalenv['dbn_inference']
plot_gt_funtion_r = ro.globalenv['plot_ground_truths']

# Imports for Deep-RL using CleanRL
import argparse
import os
import random
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNCRL:
    def __init__(self, env, num_envs, screen_width, screen_height, learning_rate,
                 buffer_size, gamma, target_network_update_rate,
                 target_network_update_frequency, batch_size, start_e, end_e, exploration_fraction,
                 learning_start, train_frequency):

        # Environment
        self.name = env.spec.id  # Gym ID of the Atari game
        self.screen_width = screen_width  # Width of the game screen
        self.screen_height = screen_height  # Height of the game screen

        self.env = env
        self.num_envs = num_envs
        self.states = env.unwrapped.states
        self.actions = env.unwrapped.actions
        self.reward_variable_values = env.unwrapped.reward_variable_values
        self.reward_variable_categories = env.unwrapped.reward_variable_categories
        # Counting the number of relational states
        self.relational_states_count = 1
        for i in env.unwrapped.state_variables_cardinalities:
            self.relational_states_count *= i
        # Creating a dic to count the number of times each action is performed in a given original state
        self.original_action_count = np.zeros((len(self.states), len(self.actions)))
        # Creating a dic to count the number of times each action is performed in a given relational state
        self.relational_action_count = np.zeros((len(self.states), len(self.actions)))

        #  Algorithm
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

        # TRY NOT TO MODIFY: seeding
        self.seed_value = 1
        self.capture_video = False
        # random.seed(self.seed_value)
        # np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        torch.backends.cudnn.deterministic = True
        
        cuda = True and torch.cuda.is_available()
        num_gpus = torch.cuda.device_count()

        device_name = "cpu"
        if cuda and num_gpus == 1:
            device_name = "cuda:0"
        elif cuda and num_gpus == 2:
            device_name = "cuda:1"
        self.device = torch.device(device_name)

        # env setup
        self.envs = gym.vector.SyncVectorEnv(
            [self.make_env( i, self.capture_video, self.name) for i in
             range(self.num_envs)]
        )
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.q_network = QNetwork(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.target_network = QNetwork(self.envs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        self.global_step = 0

    def make_env(self, idx, capture_video, run_name):
        def thunk():
            # the original env
            env = self.env

            # Add some wrappers to the environment
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

            env = gym.wrappers.RecordEpisodeStatistics(env)
            if env.render_mode in ["rgb_array", "preloaded_color"]:
                env = gym.wrappers.ResizeObservation(env, (84, 84))
                env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(self.seed_value)

            return env

        return thunk


    def crl_learn(self, max_ep):

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
                                actual_episodes, step)

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
                                step_length_in_episodes, actual_episodes, step, causal_models)
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
                                step_length_in_episodes, actual_episodes, step)

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

    def learn(self, total_episodes, initial_epsilon_index, step_name, causal_models=None):
        """ Perform Deep-RL learning, return episode_reward, episode_steps, and record of rl_data """

        episode_reward = []  # cumulative reward
        steps_per_episode = []  # steps per episodes
        episode_stage = []  # to store the corresponding stage on each episode
        episode_stage.append((step_name, initial_epsilon_index, initial_epsilon_index + total_episodes))

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

        # RUN
        # For each episode
        for episode in range(total_episodes):

            obs, info = self.envs.reset(seed=self.seed_value, options={'state_index': episode, 'state_type': "original"})
            start_state = info['integer_state']

            # self.env.render()
            self.states = self.env.states  # Updating the state list in relational representation

            steps = 0
            cumulative_reward = 0
            done = False

            while steps < max_steps_per_episode and not done:

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
                            if random.random() < epsilon_values[initial_epsilon_index + episode]:  # Explore
                                a = np.random.choice(action_indexes)
                                actions = np.array([a])
                                # a = self.actions.index(np.random.choice(self.actions))
                                # a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],action_indexes))
                            else:  # Exploit selecting the best action according Q
                                #a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                                q_values = self.q_network(torch.Tensor(obs).to(self.device))
                                actions = torch.argmax(q_values, dim=1).cpu().numpy()
                                a = actions[0]
                        # elif crl_action_selection_strategy == ActionSelectionStrategy.NEW_IDEA:
                        #
                        #     # If we know the model is good, we can set the Q-values for non-filtered actions to negative
                        #     first = np.arange(len(self.actions))
                        #     second = np.array(action_indexes)
                        #     diff = np.setdiff1d(first, second)
                        #     self.q[s, diff] = -1000.0
                        #
                        #     # Then, use the same epsilon-greedy policy to select among the filtered actions_indexes
                        #     if random.random() < self.epsilon_values[initial_epsilon_index + episode]:  # Explore
                        #         a = np.random.choice(action_indexes)
                        #         # a = self.actions.index(np.random.choice(self.actions))
                        #         # a = np.random.choice(np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],action_indexes))
                        #     else:  # Exploit selecting the best action according Q
                        #         # a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                        #         a = np.random.choice(
                        #             np.intersect1d(np.where(self.q[s] == np.max(self.q[s, action_indexes]))[0],
                        #                            action_indexes))

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
                        if random.random() < epsilon_values[initial_epsilon_index + episode]:
                            # Explore but considering the best action for CD

                            candidates_actions = np.where(action_count[state_index] < min_frequency)

                            if candidates_actions[0].size > 0:
                                a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])
                                actions = np.array([a])
                                # a = np.random.choice(candidates_actions[0])
                            else:
                                a = self.actions.index(np.random.choice(self.actions))
                                actions = np.array([a])

                            # a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])
                            # candidates_actions = np.where(self.action_count[rl_index] < min_frequency)
                            #
                            # if candidates_actions[0].size > 0:
                            #     a = np.random.choice(candidates_actions[0])
                            #
                            # else:
                            #     a = self.actions.index(np.random.choice(self.actions))

                        else:  # Exploit selecting the best action according Q
                            #a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                            q_values = self.q_network(torch.Tensor(obs).to(self.device))
                            actions = torch.argmax(q_values, dim=1).cpu().numpy()
                            a = actions[0]

                    # elif model_discovery_strategy == ModelDiscoveryStrategy.NEW_IDEA:
                    #     # Aqui una idea puede ser hacer siempre acciones que considero importantes para
                    #     # CD independientemente de epsilon y de lo que diga Q.
                    #     a = np.random.choice(np.where(action_count[state_index] == np.min(action_count[state_index]))[0])
                    #
                    # elif model_discovery_strategy == ModelDiscoveryStrategy.RANDOM:
                    #     a = self.actions.index(np.random.choice(self.actions))

                    elif model_discovery_strategy == ModelDiscoveryStrategy.EPSILON_GREEDY:
                        if random.random() < epsilon_values[initial_epsilon_index + episode]:  # Explore
                            a = self.actions.index(np.random.choice(self.actions))
                        else:  # Exploit
                            #a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                            q_values = self.q_network(torch.Tensor(obs).to(self.device))
                            actions = torch.argmax(q_values, dim=1).cpu().numpy()
                            a = actions[0]

                    else: # Model discovery strategy is to select the same action always
                        a = model_discovery_strategy.value

                if step_name == Step.RL or do_rl:

                    do_rl = False

                    if rl_action_selection_strategy == ActionSelectionStrategy.EPSILON_GREEDY:
                        if random.random() < epsilon_values[initial_epsilon_index + episode]:
                            # Explore
                            # a = self.actions.index(np.random.choice(self.actions))
                            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
                            a = actions[0]
                        else:  # Exploit
                            #a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])
                            q_values = self.q_network(torch.Tensor(obs).to(self.device))
                            actions = torch.argmax(q_values, dim=1).cpu().numpy()
                            a = actions[0]

                # Update the action_count variable
                #if step_name == Step.RL_FOR_CD:
                action_count[state_index, a] += 1

                # Take action, observe outcome
                #observation, reward, terminated, truncated, info = self.env.step(a)
                #try:
                next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
                #except:
                #    next_obs, rewards, terminated, truncated, infos = self.envs.step(np.array([a]))

                done = terminated[0] or truncated[0]
                self.global_step = self.global_step + 1

                # TRY NOT TO MODIFY: record rewards and other metrics for plotting purposes
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        # Skip the envs that are not done
                        if "episode" not in info:
                            continue
                        # print(f"global_episode={self.global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                        writer.add_scalar("charts/epsilon", epsilon_values[initial_epsilon_index + episode], self.global_step)

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                for idx, d in enumerate(truncated):
                    if d:
                        real_next_obs[idx] = infos["final_observation"][idx]
                #try:
                self.rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
                #except:
                #    self.rb.add(obs, real_next_obs, np.array([a]), rewards, terminated, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                s_prime = infos["integer_state"][0]

                # Save the info for causal discovery
                record[self.actions[a]]["all_states_i"].append(self.states[s])
                record[self.actions[a]]["all_states_j"].append(self.states[s_prime])
                record[self.actions[a]]["all_rewards"].append(
                    self.reward_variable_categories[self.reward_variable_values.index(rewards[0])])

                steps = steps + 1
                cumulative_reward = cumulative_reward + rewards[0]

                # ALGO LOGIC: training.
                if self.global_step > self.learning_start:
                    if self.global_step % self.train_frequency == 0:
                        data = self.rb.sample(self.batch_size)
                        with torch.no_grad():
                            target_max, _ = self.target_network(data.next_observations).max(dim=1)
                            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                        loss = F.mse_loss(td_target, old_val)

                        if self.global_step % 100 == 0:
                            writer.add_scalar("losses/td_loss", loss, self.global_step)
                            writer.add_scalar("losses/q_values", old_val.mean().item(), self.global_step)
                            #print("SPS:", int(self.global_step / (time.time() - start_time)))
                            writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)

                        # optimize the model
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # update target network
                    if self.global_step % self.target_network_update_frequency == 0:
                        for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                                         self.q_network.parameters()):
                            target_network_param.data.copy_(
                                self.target_network_update_rate * q_network_param.data + (1.0 - self.target_network_update_rate) * target_network_param.data
                            )

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

                if steps != max_steps_per_episode:
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
        env = gym.make(environment_name, render_mode = "preloaded_color", env_type = environment_type, reward_type = reward_type, render_fps=64)

        # Params for the experiment output related folder and names
        results_folder = "Deep RL experiments"
        # Sub folders
        rl_result_folder = "rl_results"
        causal_discovery_data_folder = "cd_data_and_results"

        # A folder to store the Ground Truth models
        ground_truth_models_folder = "ground_truth_models/" + environment_name
        # Calling the R function to plot and save the Ground Truth Models
        # plot_gt_funtion_r(env.spec.name, environment_type, ground_truth_models_folder)

        # Initializing experiment related params
        experiment_name = experiment_conf.exp_name
        evaluation_metric = experiment_conf.evaluation_metric
        max_episodes = experiment_conf.max_episodes
        max_steps_per_episode = experiment_conf.max_steps
        action_count_strategy = experiment_conf.action_count_strategy
        shared_initial_states = experiment_conf.shared_initial_states
        trials = experiment_conf.trials
        smooth = experiment_conf.smooth
        number_of_algorithms = len(experiment_conf.alg_confs)

        print_info = True
        # get the start time
        st = time.time()

        experiment_folder_name = util.get_experiment_folder_name(experiment_name, env.spec.name, environment_type, max_episodes, max_steps_per_episode, action_count_strategy,shared_initial_states, trials)
        print("Starting Experiment: " + experiment_folder_name)

        # Variables to store the results on RL policy learning (reward and steps) among trials for each algorithm
        algorithm_names = [""] * number_of_algorithms
        algorithm_r = np.zeros((number_of_algorithms, trials, max_episodes))
        algorithm_steps = np.zeros((number_of_algorithms, trials, max_episodes))

        # Variable to measure CD results of each algorithm performing CD among trials
        causal_qlearning_shd_distances = []  # To store the shd_distances dict for each trial
        # To measure action selection using CM
        causal_qlearning_actions_by_model_count = []
        causal_qlearning_good_actions_by_model_count = []

        for t in range(trials):
            print("\nStarting Trial: " + str(t + 1))
            experiment_sub_folder_name = experiment_folder_name + "/trial " + str(t + 1)

            writer = SummaryWriter(f"{results_folder}/{experiment_sub_folder_name}/tensorboard_run_info/")

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

                screen_width = algorithm.screen_width  # Width of the game screen
                screen_height = algorithm.screen_height  # Height of the game screen

                #  Algorithm
                learning_rate = algorithm.learning_rate  # The learning rate of the algorithm
                buffer_size = algorithm.buffer_size  # The size of the replay memory buffer
                gamma = algorithm.gamma  # The discount factor gamma
                target_network_update_rate = algorithm.target_network_update_rate  # The target network update rate
                target_network_update_frequency = algorithm.target_network_update_frequency  # The frequency of target network updates
                batch_size = algorithm.batch_size  # The batch size for training
                start_e = algorithm.start_e  # The starting epsilon for exploration
                end_e = algorithm.end_e  # The ending epsilon for exploration
                exploration_fraction = algorithm.exploration_fraction  # The fraction of total time steps for epsilon decay
                learning_start = algorithm.learning_start  # The time step to start learning
                train_frequency = algorithm.train_frequency  # The frequency of training

                rl_action_selection_strategy = algorithm.rl_action_selection_strategy
                episode_state_initialization = algorithm.episode_state_initialization


                # Parameters for Causal-RL
                if isinstance(algorithm, CARLDQNConf):
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

                # Precalculating the epsilon values for all episodes
                epsilon_values = []
                for epi in range(max_episodes):
                    epsilon_values.append(
                    linear_schedule(start_e, end_e, exploration_fraction * max_episodes, epi))

                # Reset the environment before to start, this can be always original because we are goin to restart again later
                env.reset(options={'state_index': 0, 'state_type': "original"})

                print("\nStarting algorithm {}. {}".format(a+1, alg_name))

                # Initializing the agent
                agent = DQNCRL(env, 1, screen_width, screen_height, learning_rate, buffer_size, gamma,
                               target_network_update_rate, target_network_update_frequency, batch_size, start_e, end_e, exploration_fraction,
                               learning_start, train_frequency)

                # Check if algorithm is CRL first because CRL extend RL
                if isinstance(algorithm, CARLDQNConf):
                    # Do Deep Reinforcement Learning with CARL
                    start_time = time.time()
                    algorithm_r[a][t], algorithm_steps[a][
                        t], record, actions_by_model_count, good_actions_by_model_count, shd_distances, episode_stage = agent.crl_learn(
                        max_episodes)
                    alg_shd_distances.append(shd_distances)

                    for stage in episode_stage:
                        if stage[0] == Step.CD.value:
                            alg_doing_cd_name.append(alg_name)
                            break;

                    alg_episode_stages.append(episode_stage)

                else: #isinstance(algorithm, DeepRLConf):
                    # Do traditional Deep Reinforcement Learning without using any model
                    start_time = time.time()
                    algorithm_r[a][t], algorithm_steps[a][
                        t], record, actions_by_model_count, good_actions_by_model_count, episode_stage = agent.learn(
                        max_episodes, initial_epsilon_index=0, step_name=Step.RL)

                # Save the models
                model_path = f"{results_folder}/{experiment_sub_folder_name}/tensorboard_run_info/{alg_name}.cleanrl_model"
                torch.save(agent.q_network.state_dict(), model_path)
                # print(f"model saved to {model_path}")
                # from cleanrl_utils.evals.dqn_eval import evaluate
                #
                # episodic_returns = evaluate(
                #     model_path,
                #     make_env,
                #     args.env_id,
                #     eval_episodes=10,
                #     run_name=f"{run_name}-eval",
                #     Model=QNetwork,
                #     device=device,
                #     epsilon=0.05,
                # )
                # for idx, episodic_return in enumerate(episodic_returns):
                #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

                agent.envs.close()
                writer.close()

            print()

            # Plot and save the RL results for the given trial for all algorithms. The episode stage variable only contains
            # the stages of the last algorithm to run
            util.plot_rl_results(algorithm_names, algorithm_r[:, t], algorithm_steps[:, t], None, None,
                                 results_folder + "/" + experiment_sub_folder_name + "/" + rl_result_folder, start_e,
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
                             results_folder + "/" + experiment_folder_name + "/average/" + rl_result_folder, start_e,
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
