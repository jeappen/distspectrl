import numpy as np
from src.env_wrappers.MA_learning import MA_ProductMDP


def _filter_dict(dict2filter, keys2keep):
    return {k: dict2filter[k] for k in keys2keep}


def reshape_state_for_spec(flat_state):
    return flat_state[:-1], int(flat_state[-1])


def _remove_last_dim_from_dict(dict_2_modify):
    return {a: dict_2_modify[a][:-1] for a in dict_2_modify}


class MAProductMDPLogNScaling(MA_ProductMDP):
    """ Product MDP with LogNScaling for RLLib MA
        Introduces multiple task monitors for different groupings of agents.
        This grouping introduces log(N) task monitors,
         i.e. one that works on k, 2k, 4k , upto N agents

         This should help the reward shaping process for certain predicate forms.

         Note: self.current_stage is NOT reset during self.reset(). This should keep the stage progression
                monotonic over the learning process.

    Args:
        system : System MDP (no need for reward function)
        action_dim: action space dimension for the system
        res_model : Resource_Model (optional)
        spec : TaskSpec
        min_reward (C_l) : Min possible unshaped reward
        local_reward_bound (C_u) : Max possible absolute value of local reward
                                (quantitative semantic value)

        log_n_k : (float) minimimum # of agents in the groups the predicate holds for
        log_n_factor : Scaling factor determining # of task monitors.
                        k, f*k, f^2*k, ..., N
    """

    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound,
                 res_model=None, use_shaped_rewards=True, log_n_k=2.0, log_n_factor=2):
        super().__init__(system, action_dim, spec, min_reward, local_reward_bound,
                         res_model, use_shaped_rewards)

        # NOTE: Check assumption that predicates can handle different num_agents directly.
        #       i.e. that we can re-use same monitor for different group sizes.
        #       Possible issue, the state dim changing based on number of agents in MA_spec_compiler.py

        # Counting number of stages based on the number of terms in the series {k, f*k, f^2*k, ..., N}
        _num_stages_intermediate = np.log(
            self.num_agents / log_n_k) / np.log(log_n_factor)
        self.number_of_stages = int(_num_stages_intermediate) + 1
        self.current_stage = 0
        self.log_n_k = log_n_k
        self.log_n_factor = log_n_factor
        self.agent_group_map = {}
        self.extra_state_dim = 2

        from gym import spaces
        self.action_space = spaces.Box(low=self.system.action_space.low[0], high=self.system.action_space.high[0],
                                       shape=(self.spec.total_action_dim,), dtype=np.float32)
        # New observation space has (s,v,q,current_stage) vs just (s,v,q)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(
                                                self.spec.total_state_dim + self.extra_state_dim,),
                                            dtype=np.float32)

    def _define_agent_groups(self, num_agents_per_group):
        """ Based on self.log_n_k (k) and self.log_n_factor (f) sets a group definition in self.agent_group_map

            num_agents_per_group: Desired number of agents per group

            eg: if N = 10, k = 3, f = 2
                at stage 1, (3,3,4)
                   stage 2, (6,4)
                   stage 3, (10) """

        num_groups = max(1, self.num_agents // num_agents_per_group)
        self.agent_group_map = {}
        cur_group_assignment = num_groups - 1
        cur_spots_in_group = num_agents_per_group
        for a in self.system.agents:
            self.agent_group_map[a] = cur_group_assignment
            cur_spots_in_group -= 1
            if cur_spots_in_group == 0:
                if cur_group_assignment == 0:
                    # Last group, keep adding
                    cur_spots_in_group += 1
                else:
                    cur_group_assignment -= 1
                    cur_spots_in_group = num_agents_per_group

    def reset(self):
        ma_reset_state = self.system.reset()
        for a in self.system.agents:
            ma_reset_state[a] = np.hstack((np.append(ma_reset_state[a], self.spec.init_extra_state()),
                                           np.zeros(self.extra_state_dim)))
            ma_reset_state[a][-1] = self.current_stage
        self.state = ma_reset_state
        self._define_agent_groups(self._number_of_agents_per_group())
        return self.state

    def _number_of_agents_per_group(self):
        # Limit group size to number of agents
        return min(self.num_agents, int((self.log_n_factor ** self.current_stage) * self.log_n_k))

    def _number_of_groups_this_stage(self):
        # Have at least one group
        return max(1, int(self.num_agents // self._number_of_agents_per_group()))

    def step_monitor_for_group(self, state, actions, next_state):
        """ Step monitor for group of agents individually
            Assume state and actions are filtered to a subset of agents

            state: dict of state (filtered from self.state)
            actions: dict of actions (filtered from self.actions)

            :return: True if agents in the group are at a final state
                    """

        # NOTE: Careful, can return True even if alw() not satisfied

        local_agents = []
        if self.global_mode:
            # Means monitor has global states
            # Check if current state of any agent is global
            agents_in_global_state = []
            for a in state:
                reshaped_state_for_spec = reshape_state_for_spec(state[a])
                if reshaped_state_for_spec[-1] in self.spec.gl_monitor.global_states:
                    agents_in_global_state.append(a)
                else:
                    local_agents.append(a)
            coord_reqd = len(agents_in_global_state) > 0

            if coord_reqd:
                # Imagine a step of Global Monitor if any agents in global state (DO NOT SAVE YET)
                # (Assuming no differing global states)

                global_state = list(self.concat_states(state))
                # To extract the global monitor state
                _, _, monitor_state_oly_global_agents = self.concat_states(
                    _filter_dict(state, agents_in_global_state))
                # Edit global state to step correctly using property that all agents are synced to same global state
                global_state[2] = monitor_state_oly_global_agents
                global_actions, global_trans = self.concat_actions(actions)
                # NOTE: Can do voting on global transition action here
                global_step_actions = (global_actions, global_trans)
                (global_res_reg_state, global_monitor_state) = self.spec.extra_step_global(global_state,
                                                                                           global_step_actions,
                                                                                           len(state))
        else:
            for a in state:
                if a in self.system.agents:
                    local_agents.append(a)
            # No global monitor states, local actions are gucci
            coord_reqd = False
            global_res_reg_state, global_monitor_state = None, None

        group_at_final_state = True
        for a in state:
            if a not in self.system.agents:
                continue
            # Run local updates
            reshaped_state_for_spec = reshape_state_for_spec(state[a])
            if coord_reqd and len(local_agents) == 0:
                # means all agents are in sync at same global state
                monitor_state = global_monitor_state
                # NOTE: shares ALL registers among all agents
                #       Need to only share global
                #       Problem for alw local but nothing else because here agents should be synced to same global state
                res_reg_state = global_res_reg_state[:self.spec.monitor.n_registers]
                self.state[a] = np.hstack(
                    (np.append(next_state[a], res_reg_state), monitor_state))
                # print('testglob',global_res_reg_state,res_reg_state,monitor_state)
            else:
                # Stop global state agents from picking a monitor transition
                # and wait for agents to sync
                # NOTE: this prevents global agents from doing anything while waiting
                if a in local_agents:
                    (res_reg_state, monitor_state) = self.spec.extra_step(
                        reshaped_state_for_spec, actions[a])
                    self.state[a] = np.hstack(
                        (np.append(next_state[a], res_reg_state), monitor_state))
                else:
                    # Here agent in global state keeps task monitor fixed
                    # Update registers but don't change monitor state
                    # Need register updating since there might be alw checks required
                    res_reg_state = global_res_reg_state[:self.spec.monitor.n_registers]
                    old_monitor_state = next_state[a][-1]
                    self.state[a] = np.hstack(
                        (np.append(next_state[a], res_reg_state), old_monitor_state))

            if group_at_final_state:
                monitor_state = self.state[a][-1]
                # Note: assuming global mode => self.gl_monitor defined
                if self.spec.gl_monitor.rewards[int(monitor_state)] is None:
                    group_at_final_state = False

            # Append current_stage as well
            self.state[a] = np.hstack((self.state[a], self.current_stage))

        return group_at_final_state

    def _move_to_next_stage(self):
        """ If current stage is done, move to next """
        if self.current_stage == self.number_of_stages - 1:
            # All stages done!
            return
        print("\n PROGRESS: INCREMENTING STAGE {}\n".format(self.current_stage))
        self.current_stage += 1
        self._define_agent_groups(self._number_of_agents_per_group())

    def step(self, actions):
        system_actions = {a: actions[a][:self.spec.action_dim]
                          for a in self.system.agents}
        # print(system_actions,self.spec.action_dim)
        next_state, rew, done, render = self.system.step(system_actions)
        # Only keeping the observation relevant to the current task monitor
        state_wrt_current_stage = _remove_last_dim_from_dict(self.state)
        current_stage_done = True

        for group in range(self._number_of_groups_this_stage()):
            # storing agents who are part of this group
            filtered_agents = []
            for a in self.system.agents:
                if self.agent_group_map[a] == group:
                    filtered_agents.append(a)
            filtered_state = _filter_dict(
                state_wrt_current_stage, filtered_agents)
            filtered_actions = _filter_dict(actions, filtered_agents)
            group_at_final_state = self.step_monitor_for_group(
                filtered_state, filtered_actions, next_state)
            if not group_at_final_state:
                current_stage_done = False

        state_to_return = self.state
        if current_stage_done:
            # Move to next stage but return previous stage observation for correct reward
            state_to_return = self.state.copy()
            self._move_to_next_stage()
            # Also end episode here for ease of formulating the right reward
            done = {d: True for d in done}

        info = {a: {'monitor_state': state_to_return[a][-2],
                    'render': render} for a in state_to_return}

        return state_to_return, rew, done, info


class InlineMAProductMDPLogNScaling(MAProductMDPLogNScaling):
    """ Calculates the shaped spec reward in the environment (inline)

    Does the reward scaling among different stages
    """

    def __init__(self, system, action_dim, spec, min_reward, local_reward_bound, res_model=None,
                 use_shaped_rewards=True, log_n_k=2.0, log_n_factor=2, log_n_scaling_rw_ub=1.0):
        """ Save reward scaling features

            log_n_scaling_rw_ub: Upper bound on the reward from the Task monitor
            """
        self.flat_spec = spec  # Use spec to get spec quantitative value here
        super().__init__(system, action_dim, spec, min_reward, local_reward_bound,
                         res_model, use_shaped_rewards, log_n_k, log_n_factor)

        self.state_trajectories = {}
        self.agent_ids = {"agent-" + str(i) for i in range(self.num_agents)}
        # To keep track of total trajectory reward according to quantitative spec
        self.trajectory_reward = 0
        # To keep track of total trajectory reward according to quantitative spec
        self.trajectory_rewards = {}

        # Setting k assuming a reward of the form = k*(current_stage) + task_monitor_reward
        # such that agent rewards @ stage i < rewards @ stage i+1
        self.scaling_bdd = (self.spec.max_depth + 1) * \
            local_reward_bound + log_n_scaling_rw_ub

    def concat_states(self, state_dict):
        # Consistent ordering due to self.agent_ids
        system_states = []
        reg_states = []
        monitor_states = []
        for i in range(self.num_agents):
            a = "agent-" + str(i)
            if a not in state_dict:
                continue
            reshaped_state_for_spec = reshape_state_for_spec(state_dict[a])
            sys_state, res_state, register_state, monitor_state = self.spec.extract_state_components(
                reshaped_state_for_spec)
            system_states.append(np.hstack([sys_state, res_state]))
            reg_states.append(register_state)
            monitor_states.append([monitor_state])

        return np.hstack(system_states), np.hstack(reg_states), np.hstack(monitor_states)

    def reset(self):
        ret_val = super().reset()
        self.state_trajectories = {group: [] for group in range(
            self._number_of_groups_this_stage())}
        self._add2state_trajectory(
            ret_val, self._number_of_groups_this_stage(), self.agent_group_map)
        self.trajectory_reward = 0
        return ret_val

    def _add2state_trajectory(self, state, num_groups, group_map):
        state_wrt_current_stage = _remove_last_dim_from_dict(state)
        for group in range(num_groups):
            # storing agents who are part of this group
            filtered_agents = []
            for a in self.system.agents:
                if group_map[a] == group:
                    filtered_agents.append(a)
            filtered_state = _filter_dict(
                state_wrt_current_stage, filtered_agents)
            self.state_trajectories[group].append(filtered_state)

    def step(self, actions):
        prev_stage = self.current_stage
        prev_agent_group_map = self.agent_group_map.copy()
        prev_num_groups = self._number_of_groups_this_stage()
        ret_val = super().step(actions)
        # No need below step assuming episode ends when we move up a stage
        # if prev_stage != self.current_stage:
        #     # Reset state trajectories if we move up a stage
        #     self.state_trajectories = {group: [] for group in range(self._number_of_groups_this_stage())}
        self._add2state_trajectory(
            ret_val[0], prev_num_groups, prev_agent_group_map)
        custom_reward = {agent: 0 for agent in ret_val[0]}  # Use state dict
        if self.trajectory_reward == 0:  # End of trajectory not reached yet since r/w not set
            for agent in custom_reward:
                if ret_val[2][agent]:
                    custom_reward[agent] = self.get_spectrl_reward(
                        agent, prev_agent_group_map[agent], prev_stage)
                    self.trajectory_reward = custom_reward[agent]
                    # print("Setting the r/w of agent {} to {}".format(agent,custom_reward[agent]))

        # 0 until end of trajectory
        return ret_val[0], custom_reward, ret_val[2], ret_val[3]

    def _stack_observations(self, trajectory, curr_agent_id, input_dim=2):
        """ Used to stack observations from the saved trajectory to feed into the task monitor

        If agent_id ends with a local state, order with initial agent first
        If agent_id ends with a global state, order with consistency (based on agent ID for now)
        """
        get_monitor_state = lambda x: reshape_state_for_spec(x)[-1]
        is_in_global_end = get_monitor_state(
            trajectory[-1][curr_agent_id]) in self.spec.gl_monitor.global_states

        curr_agent_i = int(curr_agent_id.split('-')[-1])
        stacked_op = [[]] * self.system.n_agents
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

        # Remove empty lists before stacking
        stacked_op = [x for x in stacked_op if len(x) > 0]
        # Stack inputs and keep current agent's register value/monitor state at end
        final_op = np.hstack(stacked_op + [curr_obs[:, input_dim:]])
        return list(map(reshape_state_for_spec, final_op))

    def get_spectrl_reward(self, agent_id, group, stage):
        """ Scale rewards based on the different stages.
            Use self.trajectory to get a value for the spec
            Current choice: +ve when all stages are done i.e. self.current_stage == self.number_of_stages - 1
                            -ve otherwise
        """
        # Each agent gets a different reward under current formulation (Markov Game).
        #  Need to enforce that here by splitting the trajectories before feeding to cum_shaped reward
        #  ALSO, global states have the concat state ordered by Agent id, local states have first state as self
        sys_dim = self.spec.state_dim
        stacked_obs = self._stack_observations(
            self.state_trajectories[group], agent_id, sys_dim)
        trajectory_reward = self.spec.cum_reward_shaped(stacked_obs, True)
        stage_reward = (stage - self.number_of_stages + 1) * self.scaling_bdd
        # Scaling to reduce reward range when num_agents and number_of_stages is large
        return (stage_reward + trajectory_reward) / self.num_agents
