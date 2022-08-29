"""Base Nav class that defines the rendering process
"""

import gym
import pdb
import itertools
from sklearn.neighbors import NearestNeighbors
from matplotlib.pyplot import gca
from os import path
import configparser
from gym.utils import seeding
from gym import spaces, error, utils
import random

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.env import MultiAgentEnv


from scipy.stats import truncnorm
from numpy import linalg as LA

import pickle

# Colors from SSD
DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

# CAR 2D from SPECTRL
# Define model of the system
# System of car in 2d with controllable velocity


class VC_Env:
    def __init__(self, time_limit, std=0.5):
        self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        self.time_limit = time_limit
        self.time = 0
        self.std = std

    def reset(self):
        self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        self.time = 0
        return self.state

    def step(self, action):
        next_state = self.state + action + truncnorm.rvs(-1, 1, 0, self.std, 2)
        self.state = next_state
        self.time = self.time + 1
        return next_state, 0, self.time > self.time_limit, None

    def render(self):
        pass


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

# From Arbaaz gym flock, deterministic and 10 agents initially


class NavEnv(MultiAgentEnv):
    """ Multiple agents moving in a Linear System with waypoints provided as observation
        Goal to distribute evenly
    """

    def __init__(self, n_agents=3, time_horizon=400, end_when_goal_reached=False, end_when_area_exited=True):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True  # if the agents are moving or not
        # normalize the adjacency matrix by the number of neighbors or not
        self.mean_pooling = False
        # self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 0
        self.n_agents = n_agents
        # number of goals, visible to all agents for now
        self.n_goals = 3
        # goal range (agents within this region have reached the goal)
        self.goal_range = 0.2
        # goal distribution : how many goals are meant to be used
        self.goals_used = self.n_goals - 1
        # Constant used in reward (fix this)
        self.goals_over_limit_penalty_const = -1

        # number states per agent
        self.nx_system = 4
        # dimension of system x-y
        self.x_dim = 2
        # number of features per agent
        self.n_features = self.nx_system  # *self.n_goals # Assuming one goal for now
        # number of actions per agent
        self.nu = 2

        # Time horizon to set at start
        self.time_horizon = time_horizon

        # To end when the goal is reached (False for custom goals)
        self.end_when_goal_reached = end_when_goal_reached

        # Maximum boundary to end trajectory
        # End run if agent goes beyond this region
        self.end_when_area_exited = end_when_area_exited
        self.x_bound = 1000

        # problem parameters from file

        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        self.max_accel = 1
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_features,),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

        self.agents = {}

    def get_a_net(self):
        return self.a_net

    def get_agent_index(self, agent):
        return self.agents.index(agent)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):

        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = action  # self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        action = np.hstack(list(agent_actions.values()))
        self.u = np.reshape(action, (self.n_agents, self.nu))

        self.x[:, 0] = self.x[:, 0] + self.u[:, 0] * 0.1
        # update  y position
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1] * 0.1

        done = False
        self.counter += 1

        # diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
        # diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        dist2goal = (x_dis ** 2 + y_dis ** 2)**0.5
        near_goal = dist2goal < self.goal_range

        if self.end_when_area_exited:
            # Check if each agent  is within the active area -x_bound,x_bound)
            done = (self.x[:, 0:2] > self.x_bound).any() or (
                self.x[:, 0:2] < -self.x_bound).any()

        if self.counter >= self.time_horizon:
            done = True
        if self.end_when_goal_reached and near_goal.any(axis=0).all():
            # Check if each agent (all) is near some goal (any)
            done = True

        # centralized reward for now (not agent-specific)
        rewards = {a: self.instant_cost() for a in self.agents}
        dones = {a: done for a in self.agents}
        dones['__all__'] = done

        return self._turn_mat_to_MA_dict(self._get_obs()), rewards, dones, {}

    def instant_cost(self):
        # Cost to reach goal
        # for even goal distribution, check environment below

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        robot_xs = self.x[:, 0]
        robot_ys = self.x[:, 1]

        robot_goalxs = self.x[:, 2]
        robot_goalys = self.x[:, 3]

        diff = ((robot_xs - robot_goalxs)**2 +
                (robot_ys - robot_goalys)**2)**0.5
        return -np.sum(diff)

        # self.feats[:,::2] = x_dis.T
        # self.feats[:,1::2] = y_dis.T

        # sum of differences in velocities

        # robot_xs = self.x[:,0]
        # robot_ys = self.x[:,1]

        # robot_goalxs = self.x[:,2]
        # robot_goalys = self.x[:,3]

        # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
        # return -np.sum(diff)

    def reset(self):
        self.agents = [f'agent-{_i}' for _i in range(self.n_agents)]

        # keep this to track position
        x = np.zeros((self.n_agents, self.nx_system))
        # this is the feature we return to the agent
        self.feats = np.zeros((self.n_agents, self.n_features))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0

        # set arbitrary goal
        self.goal_x1 = 0
        self.goal_y1 = np.random.uniform(2, 3)

        self.goal_x2 = -2
        self.goal_y2 = np.random.uniform(2, 3)

        # self.goal_x3 = -4
        # self.goal_y3 = np.random.uniform(2,3)

        # self.goal_x4 = -6
        # self.goal_y4 = np.random.uniform(2,3)

        # self.goal_x5 = -8
        # self.goal_y5 = np.random.uniform(2,3)

        self.goal_x6 = 2
        self.goal_y6 = np.random.uniform(2, 3)

        # self.goal_x7 = 4
        # self.goal_y7 = np.random.uniform(2,3)

        # self.goal_x8 = 6
        # self.goal_y8 = np.random.uniform(2,3)

        # self.goal_x9 = 8
        # self.goal_y9 = np.random.uniform(2,3)

        # self.goal_x10 = 10
        # self.goal_y10 = np.random.uniform(2,3)

        n = self.n_agents
        xpoints = np.linspace(-2 * ((n) // 2), 2 * ((n) // 2),
                              n if n % 2 else n + 1)[:self.n_agents]
        #ypoints = np.array((0,0,0,0,0))

        ypoints = np.array((np.random.uniform(-1, 0, self.n_agents)))

        self.start_xpoints = xpoints
        self.start_ypoints = ypoints

        self.goal_xpoints = np.array((self.goal_x1, self.goal_x2, self.goal_x6))[
            :self.n_goals]
        # self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5,self.goal_x6,self.goal_x7,self.goal_x8,self.goal_x9,self.goal_x10))
        self.goal_ypoints = np.array((self.goal_y1, self.goal_y2, self.goal_y6))[
            :self.n_goals]
        # self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5,self.goal_y6,self.goal_y7,self.goal_y8,self.goal_y9,self.goal_y10))

        x[:, 0] = xpoints  # - self.goal_xpoints
        x[:, 1] = ypoints  # - self.goal_ypoints

        num_goals_used = min(self.n_goals, self.n_agents)
        x[:self.n_goals, 2] = self.goal_xpoints[:num_goals_used]
        x[:self.n_goals, 3] = self.goal_ypoints[:num_goals_used]
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        return self._turn_mat_to_MA_dict(self._get_obs())

    def _turn_mat_to_MA_dict(self, matrix):
        """ Turns a matrix [n_agent* N] to the dict format for rllib Multiagent
        """
        output = {}
        for i, a in enumerate(self.agents):
            output[a] = matrix[i, :]
        return output

    def _get_obs(self):

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        if self.feats.shape[1] == 2:
            # Means using just one goal as feature
            self.feats[:, 0] = np.diag(x_dis)
            self.feats[:, 1] = np.diag(y_dis)
        if self.feats.shape[1] == self.nx_system:
            self.feats = self.x
        else:
            self.feats[:, ::2] = x_dis.T
            self.feats[:, 1::2] = y_dis.T

        return self.feats

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                       np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:, 2:4])
            a_net = np.array(neigh.kneighbors_graph(
                mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    def render(self, filename=None, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Returns a tuple of line objects, thus the comma
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')

            ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            ax.plot(self.goal_xpoints, self.goal_ypoints, 'rx')

            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('2D Navigation')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class NavEnv3D(MultiAgentEnv):
    """ Multiple agents moving in a Linear System with waypoints provided as observation
        Goal to distribute evenly
    """

    def __init__(self, n_agents=3, time_horizon=400):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True  # if the agents are moving or not
        # normalize the adjacency matrix by the number of neighbors or not
        self.mean_pooling = False
        # self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 1
        self.n_agents = n_agents
        # number of goals, visible to all agents for now
        self.n_goals = 3
        # goal range (agents within this region have reached the goal)
        self.goal_range = 0.2
        # goal distribution : how many goals are meant to be used
        self.goals_used = self.n_goals - 1
        # Constant used in reward (fix this)
        self.goals_over_limit_penalty_const = -1

        # number states per agent
        self.nx_system = 6
        # dimension of system x-y-z
        self.x_dim = 3
        # number of features per agent
        self.n_features = self.nx_system  # *self.n_goals # Assuming one goal for now
        # number of actions per agent
        self.nu = 3
        # Time horizon to set at start
        self.time_horizon = time_horizon

        # problem parameters from file

        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        self.max_accel = 1
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(self.nu,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_features,),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

        self.agents = {}

    def get_a_net(self):
        return self.a_net

    def get_agent_index(self, agent):
        return self.agents.index(agent)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):

        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = action  # self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        action = np.hstack(list(agent_actions.values()))
        self.u = np.reshape(action, (self.n_agents, self.nu))

        self.x[:, 0] = self.x[:, 0] + self.u[:, 0] * 0.1
        # update  y position
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1] * 0.1

        self.x[:, 2] = self.x[:, 2] + self.u[:, 2] * 0.1

        done = False
        self.counter += 1

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 3].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 4].reshape(1, -1).T
        z_dis = self.x[:, 2] - self.x[:self.n_goals, 5].reshape(1, -1).T
        dist2goal = (x_dis ** 2 + y_dis ** 2 + z_dis ** 2)**0.5
        near_goal = dist2goal < self.goal_range

        if self.counter >= self.time_horizon:
            done = True
        if near_goal.any(axis=0).all():  # Check if each agent (all) is near some goal (any)
            done = True

        # centralized reward for now (not agent-specific)
        rewards = {a: self.instant_cost() for a in self.agents}
        dones = {a: done for a in self.agents}
        dones['__all__'] = done

        return self._turn_mat_to_MA_dict(self._get_obs()), rewards, dones, {}

    def instant_cost(self):
        # Cost to reach goal (X-Y)
        # for even goal distribution, check environment below

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        robot_xs = self.x[:, 0]
        robot_ys = self.x[:, 1]

        robot_goalxs = self.x[:, 2]
        robot_goalys = self.x[:, 3]

        diff = ((robot_xs - robot_goalxs)**2 +
                (robot_ys - robot_goalys)**2)**0.5
        return -np.sum(diff)

    def reset(self):
        self.agents = [f'agent-{_i}' for _i in range(self.n_agents)]
        # TODO: Change this to self.nx_system (after using all old checkpoint data)
        # keep this to track position
        x = np.zeros((self.n_agents, self.nx_system))
        # this is the feature we return to the agent
        self.feats = np.zeros((self.n_agents, self.n_features))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0

        # set arbitrary goal

        self.goal_z = np.random.uniform(2, 3, self.n_agents)
        self.goal_x1 = 0
        self.goal_y1 = np.random.uniform(2, 3)

        self.goal_x2 = -2
        self.goal_y2 = np.random.uniform(2, 3)

        # self.goal_x3 = -4
        # self.goal_y3 = np.random.uniform(2,3)

        # self.goal_x4 = -6
        # self.goal_y4 = np.random.uniform(2,3)

        # self.goal_x5 = -8
        # self.goal_y5 = np.random.uniform(2,3)

        self.goal_x6 = 2
        self.goal_y6 = np.random.uniform(2, 3)

        # self.goal_x7 = 4
        # self.goal_y7 = np.random.uniform(2,3)

        # self.goal_x8 = 6
        # self.goal_y8 = np.random.uniform(2,3)

        # self.goal_x9 = 8
        # self.goal_y9 = np.random.uniform(2,3)

        # self.goal_x10 = 10
        # self.goal_y10 = np.random.uniform(2,3)

        xpoints = np.array((0, -2, 2, 4, -4, -6, -8, 6, 8, 10))[:self.n_agents]
        #ypoints = np.array((0,0,0,0,0))

        ypoints = np.array((np.random.uniform(-1, 0, self.n_agents)))
        zpoints = np.array((np.random.uniform(-1, 0, self.n_agents)))

        self.start_xpoints = xpoints
        self.start_ypoints = ypoints
        self.start_zpoints = zpoints

        self.goal_xpoints = np.array(
            (self.goal_x1, self.goal_x2, self.goal_x6))
        # self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5,self.goal_x6,self.goal_x7,self.goal_x8,self.goal_x9,self.goal_x10))
        self.goal_ypoints = np.array(
            (self.goal_y1, self.goal_y2, self.goal_y6))
        # self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5,self.goal_y6,self.goal_y7,self.goal_y8,self.goal_y9,self.goal_y10))

        x[:, 0] = xpoints  # - self.goal_xpoints
        x[:, 1] = ypoints  # - self.goal_ypoints
        x[:, 2] = zpoints  # - self.goal_ypoints

        #x[:,2] = np.array((self.goal_x1,self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5))
        #x[:,3] = np.array((self.goal_y1,self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5))

        x[:self.n_goals, 3] = self.goal_xpoints
        x[:self.n_goals, 4] = self.goal_ypoints
        x[:self.n_goals, 5] = self.goal_z
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        return self._turn_mat_to_MA_dict(self._get_obs())

    def _turn_mat_to_MA_dict(self, matrix):
        """ Turns a matrix [n_agent* N] to the dict format for rllib Multiagent
        """
        output = {}
        for i, a in enumerate(self.agents):
            output[a] = matrix[i, :]
        return output

    def _get_obs(self):

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        if self.feats.shape[1] == 2:
            # Means using just one goal as feature
            self.feats[:, 0] = np.diag(x_dis)
            self.feats[:, 1] = np.diag(y_dis)
        if self.feats.shape[1] == self.nx_system:
            self.feats = self.x
        else:
            self.feats[:, ::2] = x_dis.T
            self.feats[:, 1::2] = y_dis.T

        # displacement to just one goal below (from formation flying)
        # self.feats[:,0] = self.x[:,0] - self.x[:,2]
        # self.feats[:,1] = self.x[:,1] - self.x[:,3]

        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        # return (state_values, state_network)
        return self.feats

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                       np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:, 2:4])
            a_net = np.array(neigh.kneighbors_graph(
                mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    def render(self, filename=None, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            # plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Returns a tuple of line objects, thus the comma
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')

            #ax.plot([0], [0], 'kx')
            ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            ax.plot(self.goal_xpoints, self.goal_ypoints, 'rx')

            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('2D Navigation')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class NavEnvGym(gym.Env):
    """ Multiple agents moving in a Linear System with waypoints provided as observation
        Goal to distribute evenly
        Gym Version of Environment for GPG (Arbaaz20)
    """

    def __init__(self, n_agents=4, n_goals=3):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True  # if the agents are moving or not
        # normalize the adjacency matrix by the number of neighbors or not
        self.mean_pooling = False
        # self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 1
        self.n_agents = n_agents
        # n_agent limit for goals
        assert n_agents <= 10
        # number of goals, visible to all agents for now, NOTE: assumes goals <= a
        self.n_goals = n_goals
        # goal range (agents within this region have reached the goal)
        self.goal_range = 0.2
        # goal distribution : how many goals are meant to be used,
        # NOTE: for even distribution make divisor of n_agents
        self.goals_used = self.n_goals - 1
        self.goals_agent_limit = self.n_agents // self.goals_used
        # Constant used in reward (fix this)
        self.goals_over_limit_penalty_const = -1

        # number states per agent
        self.nx_system = 4
        # number of features per agent
        self.n_features = 2 * self.n_goals
        # number of actions per agent
        self.nu = 2

        # problem parameters from file

        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_features,),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

        self.agents = {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.u = np.reshape(action, (self.n_agents, self.nu))

        self.x[:, 0] = self.x[:, 0] + self.u[:, 0] * 0.1
        # update  y position
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1] * 0.1

        done = False
        self.counter += 1

        # diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
        # diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        dist2goal = (x_dis ** 2 + y_dis ** 2)**0.5
        near_goal = dist2goal < self.goal_range

        if self.counter > 400:
            done = True
        if near_goal.any(axis=0).all():  # Check if each agent (all) is near some goal (any)
            done = True

        # centralized reward for now (not agent-specific)

        return self._get_obs(), self.instant_cost(), done, {}

    def instant_cost(self):

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        dist2goal = (x_dis ** 2 + y_dis ** 2)**0.5
        near_goal = dist2goal < self.goal_range
        # want to distribute agents evenly among self.goals_used (limit here is akin to resource usage limit)
        goals_over_limit = (near_goal.sum(axis=1) > self.goals_agent_limit)
        # if this is zero, no goals over the limit
        goals_over_limit_penalty = sum(
            goals_over_limit) * self.goals_over_limit_penalty_const

        # reward ensuring all agents are at some goal
        reward_agent_goal = -dist2goal.min(axis=0).sum()

        # TODO: ADD COLLISION REWARD
        shaped_reward = reward_agent_goal + goals_over_limit_penalty

        return shaped_reward
        # self.feats[:,::2] = x_dis.T
        # self.feats[:,1::2] = y_dis.T

        # sum of differences in velocities

        # robot_xs = self.x[:,0]
        # robot_ys = self.x[:,1]

        # robot_goalxs = self.x[:,2]
        # robot_goalys = self.x[:,3]

        # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
        # return -np.sum(diff)

    def reset(self):
        self.agents = [f'agent-{_i}' for _i in range(self.n_agents)]
        # keep this to track position
        x = np.zeros((self.n_agents, self.n_features + 2))
        # this is the feature we return to the agent
        self.feats = np.zeros((self.n_agents, self.n_features))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0

        # set arbitrary goal
        self.goal_x1 = 0
        self.goal_y1 = np.random.uniform(2, 3)

        self.goal_x2 = -2
        self.goal_y2 = np.random.uniform(2, 3)

        self.goal_x3 = -4
        self.goal_y3 = np.random.uniform(2, 3)

        self.goal_x4 = -6
        self.goal_y4 = np.random.uniform(2, 3)

        self.goal_x5 = -8
        self.goal_y5 = np.random.uniform(2, 3)

        self.goal_x6 = 2
        self.goal_y6 = np.random.uniform(2, 3)

        self.goal_x7 = 4
        self.goal_y7 = np.random.uniform(2, 3)

        self.goal_x8 = 6
        self.goal_y8 = np.random.uniform(2, 3)

        self.goal_x9 = 8
        self.goal_y9 = np.random.uniform(2, 3)

        self.goal_x10 = 10
        self.goal_y10 = np.random.uniform(2, 3)

        xpoints = np.array((0, -2, 2, 4, -4, -6, -8, 6, 8, 10))[:self.n_agents]
        #ypoints = np.array((0,0,0,0,0))

        ypoints = np.array((np.random.uniform(-1, 0, self.n_agents)))

        self.start_xpoints = xpoints
        self.start_ypoints = ypoints

        self.goal_xpoints = np.array((self.goal_x1, self.goal_x2, self.goal_x6,
                                      self.goal_x2, self.goal_x3, self.goal_x4, self.goal_x5, self.goal_x7,
                                      self.goal_x8, self.goal_x9, self.goal_x10))[:self.n_goals]
        self.goal_ypoints = np.array((self.goal_y1, self.goal_y2, self.goal_y6,
                                      self.goal_y2, self.goal_y3, self.goal_y4, self.goal_y5, self.goal_y7,
                                      self.goal_y8, self.goal_y9, self.goal_y10))[:self.n_goals]

        x[:, 0] = xpoints  # - self.goal_xpoints
        x[:, 1] = ypoints  # - self.goal_ypoints

        #x[:,2] = np.array((self.goal_x1,self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5))
        #x[:,3] = np.array((self.goal_y1,self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5))

        x[:self.n_goals, 2] = self.goal_xpoints
        x[:self.n_goals, 3] = self.goal_ypoints
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        return self._get_obs()

    def _get_obs(self):

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        self.feats[:, ::2] = x_dis.T
        self.feats[:, 1::2] = y_dis.T

        # displacement to just one goal below (from formation flying)
        # self.feats[:,0] = self.x[:,0] - self.x[:,2]
        # self.feats[:,1] = self.x[:,1] - self.x[:,3]

        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        # return (state_values, state_network)
        return self.feats

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                       np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:, 2:4])
            a_net = np.array(neigh.kneighbors_graph(
                mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    def render(self, filename=None, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            # plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Returns a tuple of line objects, thus the comma
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')

            #ax.plot([0], [0], 'kx')
            ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            ax.plot(self.goal_xpoints, self.goal_ypoints, 'rx')

            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
