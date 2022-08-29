from spectrl.main import ProductMDP
from src.main.MA_monitor import Compiled_Spec, Resource_Model
import numpy as np
from ray.rllib.env import MultiAgentEnv
import time
import gym


def _filter_dict(dict2filter, keys2keep):
    return {k: dict2filter[k] for k in keys2keep}


def reshape_state_for_spec(flat_state):
    return flat_state[:-1], int(flat_state[-1])


class MA_ProductMDP(MultiAgentEnv):
    """ Product MDP for RLLib MA"""

    # system : System MDP (no need for reward function)
    # action_dim: action space dimension for the system
    # res_model : Resource_Model (optional)
    # spec : TaskSpec
    # min_reward (C_l) = Min possible unshaped reward
    # local_reward_bound (C_u) = Max possible absolute value of local reward (quant. sem. value)
    # multi_specs =  list of specifications (one for each agent), None if not used
    # multi_spec_map = fn(agent_id) mapping agent id to multi_spec index
    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True, multi_specs=None, multi_spec_map=None):
        self.system = system
        self.num_agents = system.n_agents
        init_system_state = self.system.reset()
        system_state_dim = len(list(init_system_state.values())[0])
        if res_model is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])

            res_model = Resource_Model(
                system_state_dim, action_dim, 0, np.array([]), delta)
        monitor = spec.get_monitor(
            system_state_dim, res_model.res_dim, local_reward_bound)
        # get global version of monitor for training
        gl_monitor = spec.get_monitor(system_state_dim, res_model.res_dim, local_reward_bound,
                                      create_global_monitor=True)

        # To prevent Wandb crashing
        time.sleep(5)

        if multi_specs is not None:
            assert(len(multi_specs) != 0)  # Good coding practices
        monitors = [_spec.get_monitor(system_state_dim, res_model.res_dim, local_reward_bound,
                                      create_global_monitor=True)
                    for _spec in multi_specs] if multi_specs is not None else None
        # To handle global monitor construction when the function might not directly have a global predicate
        # eg :seq of two global predicates
        if gl_monitor.num_agents == 1:
            gl_monitor = None
            self.global_mode = False
        else:
            self.global_mode = True

        self.spec = Compiled_Spec(res_model, monitor, min_reward, local_reward_bound, gl_monitor=gl_monitor,
                                  monitors=monitors, multispec_map=multi_spec_map)
        for a in self.system.agents:
            init_system_state[a] = (
                np.append(init_system_state[a], self.spec.init_extra_state()), 0)
        self.state = init_system_state
        self.is_shaped = use_shaped_rewards

        from gym import spaces
        self.action_space = spaces.Box(low=self.system.action_space.low[0], high=self.system.action_space.high[0],
                                       shape=(self.spec.total_action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(
                                                self.spec.total_state_dim + 1,),
                                            dtype=np.float32)
        self._agent_ids = set(self.system.agents)

    def reset(self, **kwargs):
        MA_reset_state = self.system.reset(**kwargs)
        for a in self.system.agents:
            MA_reset_state[a] = np.hstack(
                (np.append(MA_reset_state[a], self.spec.init_extra_state()), 0))
        self.state = MA_reset_state
        return self.state

    # Concat to form global state
    def concat_states(self, states):
        system_states = []
        reg_states = []
        monitor_states = []
        for a in self.system.agents:
            reshaped_state_for_spec = reshape_state_for_spec(states[a])
            sys_state, res_state, register_state, monitor_state = self.spec.extract_state_components(
                reshaped_state_for_spec)
            system_states.append(np.hstack([sys_state, res_state]))
            reg_states.append(register_state)
            monitor_states.append([monitor_state])

        return np.hstack(system_states), np.hstack(reg_states), np.hstack(monitor_states)

    # Concat to form global action and separate transition actions
    def concat_actions(self, actions):
        sys_actions = []
        trans_actions = []
        for a in actions:
            action = actions[a]
            system_action = action[:self.spec.action_dim]
            transition_action = action[self.spec.action_dim:]
            sys_actions.append(system_action)
            trans_actions.append(transition_action)

        return np.hstack(sys_actions), np.hstack(trans_actions)

    def step(self, actions):
        system_actions = {a: actions[a][:self.spec.action_dim]
                          for a in self.system.agents}
        next_state, rew, done, render = self.system.step(system_actions)
        # or 'slow' agents if one agent is in a global state
        local_agents = self.system.agents
        if self.global_mode:
            # Means monitor has global states
            # Check if current state of any agent is global
            agents_in_global_state = []
            local_agents = []
            for a in self.system.agents:
                reshaped_state_for_spec = reshape_state_for_spec(self.state[a])
                if reshaped_state_for_spec[-1] in self.spec.gl_monitor.global_states:
                    agents_in_global_state.append(a)
                else:
                    local_agents.append(a)
            coord_reqd = len(agents_in_global_state) > 0

            if coord_reqd:
                # Imagine a step of Global Monitor if any agents in global state (DO NOT SAVE YET)
                # (Assuming no differing global states)
                global_state = list(self.concat_states(self.state))
                # To extract the global monitor state from agents_in_global_state
                _, _, monitor_state_oly_global_agents = self.concat_states(
                    _filter_dict(self.state, agents_in_global_state))
                # Edit global state to step correctly using property that all agents are synced to same global state
                global_state[2] = monitor_state_oly_global_agents
                global_actions, global_trans = self.concat_actions(actions)
                # Do voting on global transition action here
                global_step_actions = (global_actions, global_trans)
                (global_res_reg_state, global_monitor_state) = self.spec.extra_step_global(global_state,
                                                                                           global_step_actions,
                                                                                           len(self.system.agents))
        else:
            # No global monitor states, local actions are gucci
            coord_reqd = False
            global_res_reg_state, global_monitor_state = None, None

        info = {}
        for a in self.system.agents:
            # Run local updates
            reshaped_state_for_spec = reshape_state_for_spec(self.state[a])
            if coord_reqd and len(local_agents) == 0:
                # means all agents are in sync at same global state
                monitor_state = global_monitor_state
                # Shares ALL registers among all agents
                #       Need to only share global
                #       Problem for alw local but nothing else because here agents should be synced to same global state
                #       Eg: if one agent violates an alw local condition, then we may forget this violation by sharing all registers.
                # HACK v1: Trying last n dim since seq update seems to use from end (instead of global_res_reg_state[:self.spec.monitor.n_registers])
                # HACK v2: np.maximum to be even more optimistic (assume we can be optimistic when dealing with global updates)
                #          This is optimistic because reward functions use minimum of register values usually (>0 means predicate is satisfied.)
                res_reg_state = np.maximum(
                    global_res_reg_state[:self.spec.monitor.n_registers], global_res_reg_state[-self.spec.monitor.n_registers:])
                self.state[a] = np.hstack(
                    (np.append(next_state[a], res_reg_state), monitor_state))
            else:
                # Stop global state agents from picking a monitor transition
                # and wait for agents to sync
                # This prevents global agents from doing anything while waiting
                if a in local_agents:
                    (res_reg_state, monitor_state) = self.spec.extra_step(
                        reshaped_state_for_spec, actions[a], agent_id=a)
                    self.state[a] = np.hstack(
                        (np.append(next_state[a], res_reg_state), monitor_state))
                else:
                    # Here agent in global state keeps task monitor fixed
                    # Update registers but don't change monitor state
                    # Need register updating since there might be alw checks required
                    res_reg_state = global_res_reg_state[:self.spec.monitor.n_registers]
                    old_monitor_state = reshaped_state_for_spec[-1]
                    self.state[a] = np.hstack(
                        (np.append(next_state[a], res_reg_state), old_monitor_state))
                    monitor_state = old_monitor_state  # to add to info dict
            info[a] = {'monitor_state': monitor_state,
                       'render': render}
        # Need consensus transition at synch states
        return self.state, rew, done, info

    def cum_reward(self, rollout, global_mode=False):
        if self.is_shaped:
            return self.spec.cum_reward_shaped(rollout, global_mode=global_mode)
        else:
            return self.spec.cum_reward_unshaped(rollout)

    def render(self):
        self.system.render()


class InlineMA_ProductMDP(MA_ProductMDP):
    """Calculates the shaped spec reward in the environment (inline)"""
    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound, res_model=None,
                 use_shaped_rewards=True, **kwargs):
        self.flat_spec = spec  # Use spec to get spec quantitative value here
        super().__init__(system, action_dim, spec, min_reward, local_reward_bound,
                         res_model, use_shaped_rewards, **kwargs)

        self.state_trajectory = []
        self.agent_ids = {"agent-" + str(i) for i in range(self.num_agents)}
        # To keep track of total trajectory reward according to quantitative spec
        self.trajectory_reward = 0

    def concat_states(self, state_dict):
        # Consistent ordering due to self.agent_ids
        system_states = []
        reg_states = []
        monitor_states = []
        for i in range(self.num_agents):
            a = "agent-" + str(i)
            reshaped_state_for_spec = reshape_state_for_spec(self.state[a])
            sys_state, res_state, register_state, monitor_state = self.spec.extract_state_components(
                reshaped_state_for_spec)
            system_states.append(np.hstack([sys_state, res_state]))
            reg_states.append(register_state)
            monitor_states.append([monitor_state])

        return np.hstack(system_states), np.hstack(reg_states), np.hstack(monitor_states)

    def reset(self, **kwargs):
        ret_val = super().reset(**kwargs)
        self.state_trajectory = []
        self._add2state_trajectory(ret_val)
        self.trajectory_reward = 0
        return ret_val

    def _add2state_trajectory(self, state):
        self.state_trajectory.append(state)

    def step(self, actions):
        ret_val = super().step(actions)
        self._add2state_trajectory(ret_val[0])
        # Use state dict
        custom_reward = {
            agent: 0 for agent in ret_val[0] if agent in self.agent_ids}
        if self.trajectory_reward == 0:  # End of trajectory not reached yet since r/w not set
            for agent in custom_reward:
                if ret_val[2][agent]:
                    custom_reward[agent] = self.get_spectrl_reward(agent)
                    self.trajectory_reward = custom_reward[agent]
                    # print("Setting the r/w of agent {} to {}".format(agent,custom_reward[agent]))

        # 0 until end of trajectory
        return ret_val[0], custom_reward, ret_val[2], ret_val[3]

    def _stack_observations(self, trajectory, curr_agent_id, input_dim=2):
        # If agent_id ends with a local state, order with initial agent first
        # If agent_id ends with a global state, order with consistency (based on agent ID for now)
        get_monitor_state = lambda x: reshape_state_for_spec(x)[-1]
        if self.global_mode:
            is_in_global_end = get_monitor_state(
                trajectory[-1][curr_agent_id]) in self.spec.gl_monitor.global_states
        else:
            is_in_global_end = False

        curr_agent_i = int(curr_agent_id.split('-')[-1])
        stacked_op = [[]] * len(trajectory[0])
        curr_obs = np.array([obs[curr_agent_id] for obs in trajectory])

        stacked_op[curr_agent_i] = curr_obs[:, :input_dim]
        for agent in trajectory[0]:
            if agent == curr_agent_id:
                continue
            batch = np.array([obs[agent] for obs in trajectory])
            other_obs = batch[:, :input_dim]
            # Use agent index to keep a consistent ordering of observations
            other_agent_i = int(agent.split('-')[-1])
            stacked_op[other_agent_i] = other_obs

        if not is_in_global_end:
            # In a local monitor state, keep current agent first
            stacked_op = [stacked_op[curr_agent_i]] + \
                stacked_op[:curr_agent_i] + stacked_op[curr_agent_i + 1:]

        # Stack inputs and keep current agent's register value/monitor state at end
        final_op = np.hstack(stacked_op + [curr_obs[:, input_dim:]])
        return list(map(reshape_state_for_spec, final_op))

    def get_spectrl_reward(self, agent_id):
        #  Each agent gets a different reward under current formulation (Markov Game).
        #  Need to enforce that here by splitting the trajectories before feeding to cum_shaped reward
        #  ALSO, global states have the concat state ordered by Agent id, local states have first state as self
        # Should use self.trajectory to get a value for the spec
        sys_dim = self.spec.state_dim
        trajectory_reward = self.spec.cum_reward_shaped(
            self._stack_observations(self.state_trajectory, agent_id, sys_dim), self.global_mode, agent_id=agent_id)

        return trajectory_reward


class MA_ProductMDP_withRandTaskMonitor(InlineMA_ProductMDP):
    """ To show importance of task monitor for synchronization between agents
        Remove monitor state from state_space
        Make task monitor transitions deterministic
        Remove task monitor action from action_space
    """

    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True):
        super().__init__(system, action_dim, spec, min_reward, local_reward_bound,
                         res_model, use_shaped_rewards)
        from gym import spaces
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(self.spec.total_state_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=self.system.action_space.low[0], high=self.system.action_space.high[0],
                                       shape=(self.spec.action_dim,), dtype=np.float32)
        extra_a = np.zeros(self.spec.extra_action_dim)
        if self.spec.extra_action_dim > 1:
            # Pick second transition (Assuming first is self loop)
            extra_a[1] = 1
        self.extra_a = extra_a

    def reset(self, **kwargs):
        return self._filter_state(super().reset(**kwargs))

    def step(self, actions):
        # Modify actions to include second transition (not self)
        for a in self.system.agents:
            actions[a] = np.concatenate([actions[a], self.extra_a])
        s, r, d, info = super().step(actions)
        return self._filter_state(s), r, d, info

    def _filter_state(self, state):
        # Remove monitor state from state
        return {agent: state[agent][:-1] for agent in state}


class MAEnvtoSA(gym.Env):
    '''
    Use navenv which is of form rllib MultiAgentEnv as an Single Agent env
    With inline spec reward (TLTL/centralized SPECTRL)

        zero_reward : True to skip calculation of r/w
    '''

    def __init__(self, ma_env, spec, zero_reward=False):
        self.MAEnv = ma_env
        self.num_agents = ma_env.n_agents
        self.flat_spec = spec  # Use spec to get spec quantitative value here

        self.state_trajectory = []
        self.agent_ids = {"agent-" + str(i) for i in range(self.num_agents)}
        # To keep track of total trajectory reward according to quantitative spec
        self.trajectory_reward = 0
        self.zero_reward = zero_reward

        from gym import spaces
        base_env = self.MAEnv
        self.action_space = spaces.Box(low=self.MAEnv.action_space.low[0], high=self.MAEnv.action_space.high[0],
                                       shape=(base_env.action_space.shape[0] * self.num_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(
                                                base_env.observation_space.shape[0] * self.num_agents,),
                                            dtype=np.float32)
        self.action_dim = self.action_space.shape[0]

    def concat_states(self, state_dict):
        # Consistent ordering due to self.agent_ids
        system_states = []
        for i in range(self.num_agents):
            a = "agent-" + str(i)
            system_states.append(np.hstack([state_dict[a]]))
        return np.hstack(system_states)

    def reset(self):
        ret_val = self.MAEnv.reset()
        self.state_trajectory = [
            self.concat_states(ret_val)]  # Only save states
        self.trajectory_reward = 0
        return self.concat_states(ret_val)

    def split_actions(self, actions):
        action_dict = {}
        state_dim = len(actions) // self.num_agents
        for i in range(self.num_agents):
            a = "agent-" + str(i)
            action_dict[a] = actions[i * state_dim:(i + 1) * state_dim]
        return action_dict

    def step(self, actions):
        ret_val = self.MAEnv.step(self.split_actions(
            actions))  # self.state, rew, done, render
        combined_states = self.concat_states(ret_val[0])
        self.state_trajectory.append(combined_states)  # Only save states
        # Modify reward here
        custom_reward = {agent: 0 for agent in ret_val[0]}  # Use state dict
        # End of trajectory not reached yet since r/w not set
        if not self.zero_reward and self.trajectory_reward == 0:
            for agent in custom_reward:
                if ret_val[2][agent]:
                    self.trajectory_reward = self.get_quant_spec_reward()
                    custom_reward[agent] = self.trajectory_reward

        single_reward = max([custom_reward[a] for a in custom_reward])
        single_done = any([ret_val[2][a] for a in ret_val[2]])
        # 0 until end of trajectory? Is end done_all?
        return combined_states, single_reward, single_done, ret_val[3]

    def get_quant_spec_reward(self):
        # Should use self.trajectory to get a value for the spec
        trajectory_reward = 0
        # sys_traj and resource_traj(not used)
        trajectory_reward = self.flat_spec.quantitative_semantics(self.state_trajectory,
                                                                  self.state_trajectory[0].shape[0], True)
        return trajectory_reward


class SA_ProductMDP(ProductMDP):
    """Single Agent view of a MultiAgent System""" 
    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True):
        super().__init__(system, action_dim, spec, min_reward, local_reward_bound,
                         res_model, use_shaped_rewards)

        print("Num Monitor States: {}".format(self.spec.monitor.n_states))
        # To prevent Wandb crashing
        time.sleep(5)

        self.state_trajectory = []
        # To keep track of total trajectory reward according to quantitative spec
        self.trajectory_reward = 0

        from gym import spaces
        self.action_space = spaces.Box(low=self.system.action_space.low[0], high=self.system.action_space.high[0],
                                       shape=(self.spec.total_action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(
                                                self.spec.total_state_dim + 1,),
                                            dtype=np.float32)
        self.num_agents = system.num_agents

    # Wrappers flatten the state for ease of use in RLLib
    def reset(self):
        retval = np.hstack(super().reset())
        self.state_trajectory = [retval]
        self.trajectory_reward = 0  # Flag to set sparse reward and work w/ traj. reward
        return retval

    def step(self, action):
        # Use cum shaped reward
        state, rew, done, render = super().step(action)
        self.state_trajectory.append(state)  # Only save states
        # End of trajectory not reached yet since r/w not set
        if self.trajectory_reward == 0 and done:  # set reward and end
            self.trajectory_reward = self.cum_reward(self.state_trajectory)

        info = {'render': render,
                'monitor_state': state[-1]}

        return np.hstack(state), self.trajectory_reward, done, info
