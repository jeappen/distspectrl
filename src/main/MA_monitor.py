import numpy as np
import math

N_INF = -10000000

# Model for resources


class Resource_Model:

    # state_dim : int
    # action_dim : int
    # res_dim : int
    # res_init : np.array(res_dim)
    # res_delta : np.array(state_dim), np.array(res_dim), np.array(action_dim) -> np.array(res_dim)
    def __init__(self, state_dim, action_dim, res_dim, res_init, res_delta):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.res_dim = res_dim
        self.res_init = res_init
        self.res_delta = res_delta

# ==================================================================================================


# Class for representing monitors
class Monitor_Automaton:

    # n_states : int
    # n_registers : int
    # input_dim : int (state_dim + res_dim)
    # init_registers : np.array(n_registers)
    # transitions : adjacency list of transitions of the form (q,p,u) where,
    #               q : int (monitor state)
    #               p : np.array(input_dim) , np.array(n_registers) -> (Bool,Float {quant. sym.})
    #               u : np.array(input_dim) , np.array(n_registers) -> np.array(n_registers)
    # rewards : list of n_states reward functions for final states (others are None)
    #           rewards[i] : np.array(input_dim) , np.array(n_registers) -> Float
    # num_agents: to handle multi agent state inputs
    # global_states: list of state indexes with global predicates
    def __init__(self, n_states, n_registers, input_dim, init_registers, transitions, rewards,
                 num_agents=1, global_states=None, task_info=None):

        self.n_states = n_states
        self.n_registers = n_registers
        self.input_dim = input_dim
        self.init_registers = init_registers
        self.transitions = transitions
        self.rewards = rewards
        self.num_agents = num_agents
        if global_states is None:
            global_states = []
        self.global_states = global_states
        self.task_info = task_info

# ==================================================================================================
# Useful functions on monitors


def find_state_depths(monitor):
    depths = [0] * monitor.n_states
    incoming = [0] * monitor.n_states

    for q1 in range(monitor.n_states):
        for (q2, _, _) in monitor.transitions[q1]:
            if q1 != q2:
                incoming[q2] += 1

    topo_sort = []
    for q in range(monitor.n_states):
        if incoming[q] == 0:
            topo_sort.append(q)
            depths[q] = 0

    for i in range(monitor.n_states):
        for (q, _, _) in monitor.transitions[topo_sort[i]]:
            if q != topo_sort[i]:
                incoming[q] -= 1
                depths[q] = max(depths[q], depths[topo_sort[i]]+1)
                if incoming[q] == 0:
                    topo_sort.append(q)

    return depths

# ==================================================================================================
# Compiled Specification consisting of resource model and reward monitor.


class Compiled_Spec:

    # resource : Resource_Model
    # monitor : Monitor_Automaton
    # min_reward : float
    # local_reward_bound : float (C)
    def __init__(self, resource, monitor, min_reward, local_reward_bound,
                 gl_monitor=None, monitors=None, multispec_map=None):

        # MA relevant
        self.gl_monitor = gl_monitor

        self.state_dim = resource.state_dim
        self.action_dim = resource.action_dim

        # Resource
        self.resource = resource

        # Monitor
        if gl_monitor is not None:
            # Trying to fix mixed choice bug by having one composite task monitor
            self.monitor = gl_monitor
        else:
            self.monitor = monitor
        self.monitors = monitors  # for multispec
        self.multispec_map = multispec_map
        self.extra_action_dim = max(map(len, self.monitor.transitions))
        self.depths = find_state_depths(self.monitor)
        self.max_depth = max(self.depths)

        # Product MDP
        self.total_state_dim = self.state_dim + \
            self.resource.res_dim + self.monitor.n_registers
        self.total_action_dim = self.action_dim + self.extra_action_dim
        self.min_reward = min_reward
        self.local_reward_bound = local_reward_bound

        self.register_split = self.state_dim + self.resource.res_dim
        self.id = ""  # HACK: For compat with latest gym

    @property
    def is_global_mode(self):
        return self.gl_monitor is not None

    def _get_multispec_monitor(self, agent_id=None):
        monitor = self.monitor
        if self.monitors is not None and agent_id is not None and len(self.monitors) != 0:
            monitor = self.monitors[self.multispec_map(agent_id)]
        return monitor

    # Extract state components
    # state : (np.array(total_state_dim), int)
    def extract_state_components(self, state, global_mode=False):
        if global_mode:
            # change to handle global state where state_dim = N*S_a
            # Sorted by agent index unless in local state
            state_dim = self.gl_monitor.input_dim
            # update register_split, self.state_dim has S_a dim
            register_split = self.register_split - self.state_dim + state_dim
            if len(state[0][register_split:]) != self.gl_monitor.n_registers:
                # HACK: Means this is a subset of agent states
                register_split = len(state[0]) - self.gl_monitor.n_registers
                # Note: Ignores resource dim
                state_dim = register_split
        else:
            state_dim = self.state_dim
            register_split = self.register_split
        sys_state = state[0][:state_dim]
        res_state = state[0][state_dim:register_split]
        register_state = state[0][register_split:]
        monitor_state = state[1]

        return sys_state, res_state, register_state, monitor_state

    # Step function for Resource, Monitor state and Registers
    # state : (np.array(total_state_dim), int)
    # action : np.array(total_action_dim) or np.array(action_dim)
    # global_state: TODO  to use for global predicates?
    # return value : (np.array(resource.res_dim + monitor.n_registers), int)
    # agent_id : To be passed for accessing multispec monitors
    def extra_step(self, state, action, global_state={}, agent_id=None):
        sys_state, res_state, register_state, monitor_state = self.extract_state_components(
            state)

        sys_action = action[:self.action_dim]

        # To enable Multispec (different monitors for different agents)
        monitor = self._get_multispec_monitor(agent_id)

        # Step the Resource state
        new_res_state = np.array([])
        if self.resource.res_dim > 0:
            new_res_state = self.resource.res_delta(
                sys_state, res_state, sys_action)

        # Step the Monitor and Register state
        # Assumes at-least one predicate of the outgoing edges will be satisfied
        # start with an edge and explore to find the edge that is feasible
        edge = 0

        # if monitor state is global, need to do coordination
        # Coordination means, transitions to global monitor states happen at once
        if len(action) < self.total_action_dim:
            # pick self transition
            q, _, _ = monitor.transitions[monitor_state][0]
            if q == monitor_state:
                edge = 1
            else:
                edge = 0
        else:
            mon_action = action[self.action_dim:]
            edge = np.argmax(
                mon_action[:len(monitor.transitions[monitor_state])])

        outgoing_transitions = monitor.transitions[monitor_state]
        sys_res_state = np.concatenate([sys_state, res_state])
        for i in range(len(outgoing_transitions)):
            (q, p, u) = outgoing_transitions[(
                i+edge) % len(outgoing_transitions)]
            satisfied, _ = p(sys_res_state, register_state)
            if satisfied:
                new_register_state = u(sys_res_state, register_state)
                return (np.concatenate([new_res_state, new_register_state]), q)

    # TRANSITION voting when multiple agents pick different transitions
    #   and we wish to pick one global transition
    # num_transitions : # of transitions to pick between
    def trans_voting(self, transitions, num_transitions, num_agents):
        from collections import Counter
        reshaped_transitions = np.reshape(
            transitions, (num_agents, -1))[:, :num_transitions]
        agent_votes = reshaped_transitions.argmax(axis=1)

        # Pick most common transition (majority vote)
        cnt = Counter(agent_votes)
        most_common_count = cnt.most_common()[0][1]
        # For consistently picking state when same votes
        our_pick = min([a for a in cnt if cnt[a] == most_common_count])
        # our_pick = .most_common()[0][0]
        return our_pick

    # Step function for Global Resource, Global Monitor state and Global Registers
    # state : (np.array(total_state_dim), int)
    # action : np.array(total_action_dim) or np.array(action_dim)
    # return value : (np.array(resource.res_dim + monitor.n_registers), int)
    def extra_step_global(self, states, actions, num_agents):
        system_states, reg_states, monitor_states = states
        (global_actions, global_trans) = actions
        # HACK: assuming shared registers for global and TODO: keep local registers with the same value
        register_state = reg_states

        # NOTE: Assumes all monitor states are synced here
        monitor_state = monitor_states[0]

        # NOTE: Ignores the Resource state (not using in our implementation)

        new_res_state = np.array([])

        # Step the Monitor and Register state
        # Assumes at-least one predicate of the outgoing edges will be satisfied
        # start with an edge and explore to find the edge that is feasible
        edge = 0

        # if monitor state is global, need to do coordination
        # Coordination means, transitions to global monitor states happen at once
        if len(global_trans) < self.total_action_dim:
            # pick self transition
            q, _, _ = self.gl_monitor.transitions[monitor_state][0]
            if q == monitor_state:
                edge = 1
            else:
                edge = 0
        else:
            edge = self.trans_voting(global_trans,
                                     len(
                                         self.gl_monitor.transitions[monitor_state]),
                                     num_agents)

        outgoing_transitions = self.gl_monitor.transitions[monitor_state]
        sys_res_state = system_states
        for i in range(len(outgoing_transitions)):
            (q, p, u) = outgoing_transitions[(
                i+edge) % len(outgoing_transitions)]
            satisfied, _ = p(sys_res_state, register_state)
            if satisfied:
                new_register_state = u(sys_res_state, register_state)
                return (np.concatenate([new_res_state, new_register_state]), q)

    # Initial values of the extra state space
    # return value : np.array(resource.res_dim + monitor.n_registers)
    def init_extra_state(self):
        return np.concatenate([self.resource.res_init, self.monitor.init_registers])

    # Alpha max for a state with max q monitor state
    # sys_res_state : np.array(state_dim + resource.res_dim)
    # monitor_state : int
    # register_state : np.array(monitor.n_registers)
    def alpha_max_reward_q(self, sys_res_state, register_state, monitor_state):
        rew = N_INF
        max_q = -1
        for (q, p, u) in self.monitor.transitions[monitor_state]:
            if q == monitor_state:
                continue
            _, edge_rew = p(sys_res_state, register_state)
            if edge_rew > rew:
                max_q = q
                rew = edge_rew
        return rew, max_q

    # Alpha max for a state
    # sys_res_state : np.array(state_dim + resource.res_dim)
    # monitor_state : int
    # register_state : np.array(monitor.n_registers)
    def alpha_max_reward(self, sys_res_state, register_state, monitor_state):
        return self.alpha_max_reward_q(sys_res_state, register_state, monitor_state)[0]

    # Get alpha for a state to next_monitor_state, N_INF if not possible
    # sys_res_state : np.array(state_dim + resource.res_dim)
    # monitor_state : int
    # register_state : np.array(monitor.n_registers)
    # next_monitor_state: int
    def alpha_value(self, sys_res_state, register_state, monitor_state, next_monitor_state):
        rew = N_INF
        for (q, p, u) in self.monitor.transitions[monitor_state]:
            if q == next_monitor_state:
                _, edge_rew = p(sys_res_state, register_state)
                rew = max(rew, edge_rew)
        return rew

    # Shaped reward for a state
    # sys_res_state : np.array(state_dim + resource.res_dim)
    # monitor_state : int
    # register_state : np.array(monitor.n_registers)
    def shaped_reward(self, sys_res_state, register_state, monitor_state, global_mode=False):
        rew = N_INF
        monitor = self.gl_monitor if global_mode else self.monitor
        for (q, p, u) in monitor.transitions[monitor_state]:
            if q == monitor_state:
                continue
            _, edge_rew = p(sys_res_state, register_state)
            rew = max(rew, edge_rew)
        return self.min_reward \
            + rew \
            + (self.local_reward_bound) * \
            (self.depths[monitor_state] - self.max_depth)

    # Cumulative reward for a rollout (shaped)
    # rollout : [(np.array, int)]
    # agent_id : (str) Used to give different rewards to each agent for multispec
    def cum_reward_shaped(self, rollout, global_mode=False, agent_id=None):
        last_state = rollout[len(rollout)-1]

        monitor = self.gl_monitor if global_mode else self._get_multispec_monitor(
            agent_id)

        # Final State
        if monitor.rewards[last_state[1]] is not None:
            (last_sys_state, last_res_state, last_register_state,
             last_monitor_state) = self.extract_state_components(last_state,
                                                                 global_mode=global_mode)
            last_sys_res_state = np.concatenate(
                [last_sys_state, last_res_state])
            return monitor.rewards[last_monitor_state](last_sys_res_state, last_register_state)

        # Non-final state
        rew = N_INF
        for state in rollout:
            if(state[1] == last_state[1]):
                (sys_state, res_state, register_state,
                 monitor_state) = self.extract_state_components(state,
                                                                global_mode=global_mode)
                sys_res_state = np.concatenate([sys_state, res_state])
                rew = max(rew, self.shaped_reward(sys_res_state, register_state, monitor_state,
                                                  global_mode=global_mode))

        return rew

    # Cumulative reward for a rollout (unshaped)
    # rollout : [(np.array, int)]
    def cum_reward_unshaped(self, rollout):
        last_state = rollout[len(rollout)-1]

        # Final State
        if self.monitor.rewards[last_state[1]] is not None:
            (last_sys_state, last_res_state, last_register_state,
             last_monitor_state) = self.extract_state_components(last_state)
            last_sys_res_state = np.concatenate(
                [last_sys_state, last_res_state])
            return self.monitor.rewards[last_monitor_state](last_sys_res_state, last_register_state)

        # Non-final state
        return self.min_reward - self.local_reward_bound
