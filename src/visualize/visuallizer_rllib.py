"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

from pandas.errors import EmptyDataError
import json
import numpy as np
import os

from ray.cloudpickle import cloudpickle

import glob
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd

import json
import numpy as np
import os
import shutil
import sys


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + '/params.json'  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + '/params.pkl'  # params.json is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)
    return pkldata


x_lim_list = {'phi_2': (0, 10000), 'phi_3': (0, 10000), 'phi_1': (0, 20000)  # 20000
              # 20000
              , 'phi_4': (0, 50000), r'phi_{ex}': (0, 40000), r'phi_a': (0, 12500), 'phi_3n3_{DS}': (0, 15000), 'phi_4n3_{DS}': (0, 50000), 4: (0, 20000), 3: (0, 30000)
              }
y_lim_list = {}

depth_key_root = 'custom_metrics/final_state_reached'

y_scale = None
smoothparam = 5

# If certain plots need to be together. If match present, same plot id
# 0 means separate plots
plot_group_keys = {r'phi_2': 1, r'phi_5': 2, r'phi_3': 3, r'phi_1': 4, }
reverse_plot_group_keys = {plot_group_keys[a]: a for a in plot_group_keys}
group_plots_colour_list = ['black', 'r', 'b', 'g', 'c', 'm', 'y']
# To hack way to graph final state reached
transform_to_final_state_reached = False


def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


ax_fig_dict = {}


def plot_fig(d, spec_name, identifier="", title_addnl="", root_dir=""):
    """ 
        Args
        d: directory name w.r.t. root directory"""
    config = get_rllib_config(os.path.join(root_dir, d))
    pkl = get_rllib_pkl(os.path.join(root_dir, d))
    if 'navenv_PPO' in d:
        # Compatibility with old experiments
        depth_key_root = 'custom_metrics/max_depth_monitor_state'
    else:
        depth_key_root = 'custom_metrics/final_state_reached'
        y_scale_arg = 1
    if y_scale_arg == None:
        # Gets max monitor state that is a final state
        y_scale = np.where(pkl['env_config']['spec'].monitor.rewards)[0].max()
        if 'depth' in depth_key_root and 'monitor_state' not in depth_key_root:
            # To handle old graph keys
            # Just get max depth of a state
            y_scale = max(pkl['env_config']['spec'].depths)
            print(pkl['env_config']['spec'].depths, y_scale)
    else:
        y_scale = y_scale_arg
    progress_file = os.path.join(root_dir, d, 'progress.csv')
    cs = pd.read_csv(progress_file)
    if 'ARSTrainer' in d:
        y = (cs['episode_reward_mean'] > 0).values + 0.0  # 1 if spec satisfied
        x = (cs['info/episodes_so_far']).values
        y_max = y_min = None
    else:  # Is regular PPO
        depth_key = depth_key_root + '_mean'
        depth_key_up = depth_key_root + '_max'
        depth_key_low = depth_key_root + '_min'
        x = (cs['episodes_total']).values
        if transform_to_final_state_reached:  # HACK TO GET BOUNDS ON FINAL STATE REACHED
            # Assume last but one is incomplete
            y = cs[depth_key] - y_scale + 1
            y_max = cs[depth_key_up] - y_scale + 1
            y_min = cs[depth_key_low] - y_scale + 1
            y = np.clip(y, 0, 1)
            y_max = np.clip(y_max, 0, 1)
            y_min = np.clip(y_min, 0, 1)
        else:

            y = cs[depth_key] / y_scale
            y_max = cs[depth_key_up] / y_scale
            y_min = cs[depth_key_low] / y_scale

    # Getting the plot id if we want to group
    current_plot_id = 0
    for plot_grouping_key in plot_group_keys:
        if plot_grouping_key in spec_name:
            current_plot_id = plot_group_keys[plot_grouping_key]

    # Making the figure object
    if current_plot_id != 0:
        if current_plot_id in ax_fig_dict:
            # Plot already exists, add to it
            [fig, ax, num_plots] = ax_fig_dict[current_plot_id]
            num_plots = num_plots + 1
            ax_fig_dict[current_plot_id][-1] = num_plots
            print(num_plots, current_plot_id)
        else:
            fig, (ax) = plt.subplots(1, 1, sharex=True)
            ax_fig_dict[current_plot_id] = [fig, ax, 1]
            num_plots = 1
    else:
        fig, (ax) = plt.subplots(1, 1, sharex=True)

    if smoothparam:
        y = smooth(y, smoothparam)
    if current_plot_id == 0:
        # No comparison
        ax.plot(x, y, color='black', label='DistSPECTRL')
        facecolor_arg = (0.0, 0.5, 0.0, 0.2)
    else:
        # To group sensibly
        ax.plot(x, y, color=group_plots_colour_list[num_plots],
                label=identifier)  # '$\\'+identifier+'$')
        rgba = list(matplotlib.colors.to_rgba(
            group_plots_colour_list[num_plots]))
        rgba[-1] = .2
        facecolor_arg = rgba
    if y_max is not None:
        ax.fill_between(x, y_min, y_max,
                        where=(y_max >= y_min), facecolor=facecolor_arg, interpolate=True)
    ax.set_ylabel('Task Monitor Depth')  # 'Specification Satisfied')
    ax.set_xlabel('Number of Sampled Trajectories')

    def convert_specname_2_title(spec_name):
        parts = spec_name.split('n')
        if len(parts) == 1:  # for old unformatted spec_names
            return spec_name
        return parts[0]  # + "\ N=" + parts[1][:parts[1].index("_")]
    if current_plot_id == 0:
        ax.set_title('Specification $\\' +
                     convert_specname_2_title(spec_name) + '$' + title_addnl)
    else:
        ax.set_title('Specification $\\' +
                     reverse_plot_group_keys[current_plot_id] + '$' + title_addnl)

    # To print mean depth of last 20
    if spec_name in x_lim_list:
        xlim_ind = np.searchsorted(x, x_lim_list[spec_name][1])
        print('mean_pref', d, np.mean(y[xlim_ind - 20:xlim_ind]))
    else:
        print('mean_pref', d, np.mean(y[-20:]))

    if current_plot_id != 0:
        if current_plot_id in x_lim_list:
            ax.set_xlim(x_lim_list[current_plot_id])
    elif spec_name in x_lim_list:
        ax.set_xlim(x_lim_list[spec_name])
    if spec_name in y_lim_list:
        ax.set_ylim(y_lim_list[spec_name])

    if current_plot_id == 0:
        # Individual plots required
        ax.legend()
        fig.savefig('img/' + spec_name + '.png', dpi=300)
    else:
        ax.legend()
        fig.savefig('img/' + str(current_plot_id) + '.png', dpi=300)


def get_dir_i(i, exp_csv, only_check_long_enough_exp=True, spec_name=""):
    # To get the best hyperparameter set from a set of experiments
    # exp_csv : A pandas csv with the Exp directories
    d_to_use = None
    max_reward = -1000
    for d in glob.glob(exp_csv['Exp Name'][i] + '*/progress.csv'):
        try:
            temp_csv = pd.read_csv(d)
        except EmptyDataError:
            continue
        # Check for long enough xlim
        current_plot_id = 0
        for plot_grouping_key in plot_group_keys:
            if plot_grouping_key in spec_name:
                current_plot_id = plot_group_keys[plot_grouping_key]
        x_lim = 0
        if current_plot_id != 0:
            if current_plot_id in x_lim_list:
                x_lim = x_lim_list[current_plot_id]
        elif spec_name in x_lim_list:
            x_lim = x_lim_list[spec_name]
        if(temp_csv['episodes_total'].values[-1] < x_lim[-1]):
            continue
        cur_max = (temp_csv['episode_reward_mean']).max()
        if cur_max > max_reward:
            max_reward = cur_max
            d_to_use = d
    assert(d is not None)
    print("Using  max {} d {}".format(max_reward, d))
    return os.path.dirname(os.path.join("/", d_to_use[1:]))
