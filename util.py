import os
import numpy as np
from matplotlib import pyplot as plt
from experiments_configurations.config import Step
from pathlib import Path
from datetime import datetime
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Change this to "pdf" for the final version
export_format = [".pdf", ".svg", ".png"][2]
font_size = 16
legend_item_size = 13


def multi_plot_data(names, mean_data, std_data, folder, fig_name, y_name, x_name, legend_locs, x_fixed, episode_stage,
                    epsilon_values, legend_titles):

    linewidth = 3
    if "CARL-100%" in names:
        linewidth = 2

    fig, ax = plt.subplots(constrained_layout=True)

    lines = []

    for i in range(len(mean_data)):

        if x_fixed is not None:  # We have different sample rate at x-axis
            x = x_fixed[i]
        else:
            x = np.arange(mean_data[0].size) # All functions share the x-axis

        lines += ax.plot(x, mean_data[i], '-', linewidth=linewidth, markersize=2, label=names[i])
        if std_data is not None:
            ax.fill_between(x, mean_data[i] - std_data[i], mean_data[i] + std_data[i], alpha=0.5)

    ax.set_xlabel(x_name, fontsize=font_size)
    ax.set_ylabel(y_name, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)

    ax.legend(lines, names, fontsize=legend_item_size, loc = legend_locs[0] , title=legend_titles[0], title_fontsize=font_size)

    new_tick_locations = set()

    if episode_stage is not None:

        from matplotlib.legend import Legend

        new_lines = []

        stages_names = []

        for stage in episode_stage:
            name = stage[0]
            start = stage[1]
            end = stage[2]
            # if start == 0:
            #     new_tick_locations.add(start)
            # else:
            #     new_tick_locations.add(start - 1)
            # if start != end:
            #     new_tick_locations.add(end - 1)

            if name == Step.MODEL_INIT.value:
                # l = ax.vlines(x=start, ls='--', ymin=min(min(mean_data[0]), min(mean_data[1])),
                #               ymax=max(max(mean_data[0]), max(mean_data[1])), colors='green',
                #               label=name if name not in stages_names else "")
                new_lines.append(ax.vlines(x=start, ls='--', ymin=mean_data.min(), ymax=mean_data.max(), colors='green',
                              label=name if name not in stages_names else ""))
            elif name == Step.CD.value:
                ymin = mean_data.min()
                ymax = mean_data.max()
                if std_data is not None:
                    ymin = (mean_data - std_data).min()
                    ymax = (mean_data - std_data).max()

                new_lines.append(ax.vlines(x=start, ls='--', ymin=ymin, ymax=ymax, colors='purple',
                              label=name if name not in stages_names else ""))
            elif name == Step.RL.value:
                new_lines.append(ax.axvspan(start, end, color='blue', alpha=0.1, label=name if name not in stages_names else ""))
            elif name == Step.RL_USING_CD.value:
                new_lines.append(ax.axvspan(start, end, color='red', alpha=0.1, label=name if name not in stages_names else ""))
            elif name == Step.RL_FOR_CD.value:
                new_lines.append(ax.axvspan(start, end, color='green', alpha=0.1, label=name if name not in stages_names else ""))

            if name not in stages_names:
                stages_names.append(name)

        leg = Legend(ax, new_lines, stages_names, fontsize=legend_item_size, loc= legend_locs[1], title=legend_titles[1], title_fontsize=font_size)
        ax.add_artist(leg)
        # plt.legend(loc="lower right")
        # plt.gca().add_artist(legend1)


    if epsilon_values is not None:

        if x_fixed is not None:
            # One at the beginning
            new_tick_locations.add(x[0])
            # One at the middle
            new_tick_locations.add(int((x[len(x)-1] - x[0]) / 2))
            # One at the end
            new_tick_locations.add(x[len(x)-1])
        else: # Add a small set of point among x-axis to show the epsilon values
            # One at the beginning
            new_tick_locations.add(0)
            # One at the middle
            new_tick_locations.add(int(len(x) / 2))
            # One at the end
            new_tick_locations.add(len(x) - 1)

    # Allways plot the epsilon values axis
    ax2 = ax.twiny()

    def tick_function(X):
        return ["%.2f" % epsilon_values[z] for z in X]

    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(list(new_tick_locations), fontsize = font_size)
    ax2.set_xticklabels(tick_function(list(new_tick_locations)), fontsize = font_size)
    ax2.set_xlabel("epsilon", fontsize = font_size)

    #ax.legend(loc=legend_loc)
    if not os.path.isdir(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)

    figure = fig.savefig("{}/{}".format(folder, fig_name), bbox_inches='tight')
    plt.close(figure)


def plot_rl_results(alg_names, algorithm_r_mean, algorithm_steps_mean, algorithm_r_std, algorithm_steps_std, folder,
                 e, evaluation_metric, epsilon_values, episode_stage, smooth):

    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)

        for i in range(len(algorithm_r_mean)):
            #for j in range(len(algorithm_r_mean[i])):
            x = algorithm_r_mean[i]
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            algorithm_r_mean[i] = smoothed_x

        for i in range(len(algorithm_steps_mean)):
            #for j in range(len(algorithm_steps_mean[i])):
            x = algorithm_steps_mean[i]
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            algorithm_steps_mean[i] = smoothed_x

    multi_plot_data(alg_names, algorithm_r_mean, algorithm_r_std, folder, '%.2f' % e + "_" + evaluation_metric.value + "{}".format(export_format),
                    "avg-reward", "episodes",
                    ['center right', 'lower right'], None, episode_stage, epsilon_values, ["Algorithm", "Stage"])

    multi_plot_data(alg_names, algorithm_steps_mean, algorithm_steps_std, folder,
                    '%.2f' % e + "_" + "Episode Steps" + "{}".format(export_format),
                    "avg-steps", "episodes",
                    ['upper right', 'center right'], None, episode_stage, epsilon_values, ["Algorithm", "Stage"])

def plot_cd_results(alg_doing_cd_name, alg_shd_distances, parent_folder, alg_episode_stage, epsilon_values):

    names = list(alg_shd_distances[0].keys())

    for index in range(len(alg_doing_cd_name)):
        cd_alg_name = alg_doing_cd_name[index]
        distances = alg_shd_distances[index]
        episode_stage = alg_episode_stage[index]

        x_fixed = []

        x_values = []  # to store the CD episode numbers. Same for all actions
        for stage in episode_stage:
            if stage[0] == Step.CD.value:
                x_values.append(stage[2])

        data = []
        for action in distances:
            data.append(distances[action])
            x_fixed.append(np.array(x_values))

        multi_plot_data(names, np.array(data), None, parent_folder + "/" + cd_alg_name, "cd_analisis" + "{}".format(export_format), "shd", "episodes", ['upper right', 'lower right'],
                        x_fixed, None, epsilon_values, ["Action name", "Stage"])

    # x_fixed = []  # to store the RL using CD episode numbers
    # for stage in episode_stage:
    #     if stage[0] == Step.RL_USING_CD.value:
    #         x_fixed.append(stage[1])
    # x_fixed = np.array(x_fixed)
    #
    # data = [causal_qlearning_actions_by_model_count, causal_qlearning_good_actions_by_model_count]
    # names = ["Suggested actions", "Good actions"]
    # multi_plot_data(data, None, names, folder, '%.2f' % e + "_model_use_analisis.export_format", "count", "episodes",
    #                 'upper right',
    #                 x_fixed, None, None, None)


def plot_total_cd_results(alg_doing_cd_name, alg_total_shd_mean, alg_total_shd_std, parent_folder, alg_episode_stage, epsilon_values):

    x_fixed = []  # to store the CD episode numbers

    for index in range(len(alg_doing_cd_name)):
        episode_stage = alg_episode_stage[index]

        x_points = []
        for stage in episode_stage:
            if stage[0] == Step.CD.value:
                x_points.append(stage[2])

        x_fixed.append(np.array(x_points))

    multi_plot_data(alg_doing_cd_name, alg_total_shd_mean, alg_total_shd_std, parent_folder, "total_shd" + "{}".format(export_format), "shd", "episodes", ['upper right', 'center right'],
                        x_fixed, None, epsilon_values, ["Algorithm", "Stage"])

    # x_fixed = []  # to store the RL using CD episode numbers
    # for stage in episode_stage:
    #     if stage[0] == Step.RL_USING_CD.value:
    #         x_fixed.append(stage[1])
    # x_fixed = np.array(x_fixed)
    #
    # data = [causal_qlearning_actions_by_model_count, causal_qlearning_good_actions_by_model_count]
    # names = ["Suggested actions", "Good actions"]
    # multi_plot_data(data, None, names, folder, '%.2f' % e + "_model_use_analisis.export_format", "count", "episodes",
    #                 'upper right',
    #                 x_fixed, None, None, None)

def get_experiment_folder_name(exp_name, env, E, max_steps, action_count_strategy, shared_initial_states, trials):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d %H%M%S")
    return "{} {} {} Epi = {} steps = {} ace = {} sis = {} trials {}".format(date_time, env.spec.name, exp_name, E, max_steps,
                                                                                          action_count_strategy.value, shared_initial_states,
                                                                                          trials)

    # def save_rl_graph(self, directory_path, episode_reward, steps_per_episode):
    #     # # Average reward across trials
    #     episode_reward = np.mean(episode_reward, axis=0)
    #
    #     data = [episode_reward]
    #     names = ["Q-Learning"]
    #     multi_plot_data(data, names, directory_path + "rewards.export_format")
    #
    #     # Average steps trials
    #     steps_per_episode = np.mean(steps_per_episode, axis=0)
    #
    #     new_data = [steps_per_episode]
    #     new_names = ["Q-Learning"]
    #     multi_plot_data(new_data, new_names, directory_path + "steps.export_format")
    #
    # def save_rl_info(self, directory_path, episode_reward, steps_per_episode):
    #
    #     with open(directory_path + "/rl_info.txt",
    #               "w") as my_file:
    #         my_file.write("reward   steps" + "\n")
    #         for i in range(len(episode_reward)):
    #             my_file.write(str(episode_reward[i]) + " " + str(steps_per_episode[i]) + "\n")

    # Save RL data for the give action_name to an specific folder


def save_rl_data(folder_name, action_name, all_states_i, all_states_j, all_rewards, env, epsilon_values,
                 actual_episodes):
    file_name = action_name

    epsilon_to_string = ('%.2f' % epsilon_values[actual_episodes - 1]).replace(".", "_")
    max_episodes_to_string = str(actual_episodes)

    var_of_interest_at_time_i = ["{}{}".format(i, "I") for i in env.state_variables_names]
    var_of_interest_at_time_j = ["{}{}".format(j, "J") for j in env.state_variables_names]
    reward_variables_names = env.reward_variable_name

    var_of_interest_names = var_of_interest_at_time_i + var_of_interest_at_time_j + reward_variables_names

    heading = " ".join(var_of_interest_names)

    if not os.path.isfile(
            folder_name + "/" + epsilon_to_string + "/" + max_episodes_to_string + "/" + file_name + ".txt"):
        Path(folder_name + "/" + epsilon_to_string + "/" + max_episodes_to_string).mkdir(parents=True,
                                                                                         exist_ok=True)
        with open(folder_name + "/" + epsilon_to_string + "/" + max_episodes_to_string + "/" + file_name + ".txt",
                  "w") as my_file:
            my_file.write(heading.strip() + "\n")

    lines = []

    for i in range(len(all_states_i)):

        var_of_interest_values = []

        # First add the variables at time i
        for j in range(len(env.state_variables_names)):
            var_of_interest_values.append(str(all_states_i[i][j]))
        # Then add the variables at time j
        for j in range(len(env.state_variables_names)):
            var_of_interest_values.append(str(all_states_j[i][j]))

        # Finally add the reward
        var_of_interest_values.append(str(all_rewards[i]))

        line = " ".join(var_of_interest_values)

        lines.append(line)

    directory_path = folder_name + "/" + epsilon_to_string + "/" + max_episodes_to_string + "/"
    with open(directory_path + file_name + ".txt",
              "a") as my_file:
        my_file.writelines(l.strip() + '\n' for l in lines)
    return directory_path
