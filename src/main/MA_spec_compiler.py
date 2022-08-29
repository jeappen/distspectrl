from src.main.MA_monitor import Monitor_Automaton
from spectrl.main.compiler_utility import *

import enum

MAX_PRED_VAL = 1000000.


# Constructors for TaskSpec
class Cons(enum.Enum):
    ev = 1
    alw = 2
    seq = 3
    choose = 4
    ite = 5
    both = 6

# ==================================================================================================

# Specification AST
class TaskSpec:
    """ Modified to include global predicates in RLLib
    """

    # cons: int (Has to refer to a valid constructor in Cons)
    # predicate: np.array(state_dim), np.array(resource_dim) -> float
    # subtasks = [TaskSpec]
    # Some functions in this class assume presence of subtasks as needed,
    # according to syntax of language in the paper
    def __init__(self, cons, predicate, subtasks,
                 num_agents=1, gl_predicate=None, info=None,
                 merge_self_loop=False):
        self.cons = cons
        self.predicate = predicate
        self.subtasks = subtasks
        self.num_agents = num_agents

        # For Multi agent
        self.gl_predicate = gl_predicate
        self.info = info  # Used by the InlineMA_ProductMDP_wGoals to extract the current goal
        if len(self.subtasks) != 0:
            self.num_agents = max([st.num_agents for st in self.subtasks])

        # 'both' composition is used for centralized SPECTRL results
        # Safe to use without alw predicates
        self.merge_self_loop = merge_self_loop

    def is_global_mode(self):
        # Checks if any of the subtasks or current level task involves global predicates
        return self.gl_predicate is not None or\
            any([_st.is_global_mode() for _st in self.subtasks])

    def is_global_mode_at_current_level(self):
        # Checks if the CURRENT level predicate is global
        return self.gl_predicate is not None

    def quantitative_semantics(self, traj, sys_dim, use_prefix=False):
        sys_traj = np.array([state[:sys_dim] for state in traj])
        res_traj = [state[sys_dim:] for state in traj]
        if use_prefix:
            return self.quantitative_semantics_dp(sys_traj, res_traj)[1][len(traj)-1]
        else:
            return self.quantitative_semantics_dp(sys_traj, res_traj)[0][0]

    # Does not support conditional statements
    # Returns quantitative semantics for all suffixes (retval[0]) and prefixes (retval[1])
    # suffixes not available once a sequence operator is encountered with first task != ev
    # prefixes not available once a sequence operator is encountered with second task != ev
    # Single agent mode to use local predicates in call to quantitative_semantics_dp
    # Modified to support local predicates with input a global state trajectory

    def quantitative_semantics_dp(self, sys_traj, res_traj, single_agent_mode=False):
        n = len(sys_traj)
        retval = np.zeros((2, n))

        # atomic task
        if self.cons == Cons.ev:
            if not single_agent_mode:
                if self.num_agents > 1 and not self.is_global_mode_at_current_level():
                    # Calculate self.predicate over the split trajectory and minimize
                    per_agent_retval = []
                    for i in range(self.num_agents):
                        state_dim = len(sys_traj[0])//self.num_agents
                        retval_i = self.quantitative_semantics_dp(sys_traj[:, i*state_dim:(i+1)*state_dim],
                                                                  res_traj,  # NOTE: Ignoring resource splitting
                                                                  single_agent_mode=True)
                        per_agent_retval.append(retval_i)
                    # To combine minimize across agents?
                    for i in range(n):
                        if i == 0:
                            retval[0][n-1] = min([per_agent_retval[j][0][n-1]
                                                 for j in range(self.num_agents)])
                            retval[1][0] = min(
                                [per_agent_retval[j][1][0] for j in range(self.num_agents)])
                        else:
                            retval[0][n-i-1] = max(retval[0][n-i],
                                                   min([per_agent_retval[j][0][n-i-1] for j in range(self.num_agents)]))
                            retval[1][i] = max(retval[1][i-1],
                                               min([per_agent_retval[j][1][i] for j in range(self.num_agents)]))
                elif self.is_global_mode():
                    predicate = self.gl_predicate
                    for i in range(n):
                        if i == 0:
                            retval[0][n -
                                      1] = predicate(sys_traj[n-1], res_traj[n-1])
                            retval[1][0] = predicate(sys_traj[0], res_traj[0])
                        else:
                            retval[0][n-i-1] = max(retval[0][n-i], predicate(sys_traj[n-i-1],
                                                                             res_traj[n-i-1]))
                            retval[1][i] = max(retval[1][i-1], predicate(sys_traj[i],
                                                                         res_traj[i]))
            else:
                for i in range(n):
                    if i == 0:
                        retval[0][n -
                                  1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                        retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                    else:
                        retval[0][n-i-1] = max(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                              res_traj[n-i-1]))
                        retval[1][i] = max(retval[1][i-1], self.predicate(sys_traj[i],
                                                                          res_traj[i]))
            return retval

        # always constraint
        # this function also supports always formala without a subtask
        if self.cons == Cons.alw:
            subval = np.array([[MAX_PRED_VAL]*n]*2)
            if self.subtasks[0] is not None:
                subval = self.subtasks[0].quantitative_semantics_dp(
                    sys_traj, res_traj)
            for i in range(n):
                if i == 0:
                    retval[0][n -
                              1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                    retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                else:
                    retval[0][n-i-1] = min(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                          res_traj[n-i-1]))
                    retval[1][i] = min(retval[1][i-1], self.predicate(sys_traj[i],
                                                                      res_traj[i]))
            for i in range(n):
                retval[0][i] = min(retval[0][i], subval[0][i])
                retval[1][i] = min(retval[1][i], subval[1][i])
            return retval

        # sequence
        if self.cons == Cons.seq:
            if self.is_global_mode():
                def _get_correct_predicate(subtask):
                    if subtask.is_global_mode_locally():
                        return subtask.gl_predicate
                    else:
                        return subtask.predicate
                predicate0 = _get_correct_predicate(self.subtasks[0])
                predicate1 = _get_correct_predicate(self.subtasks[1])
            else:
                predicate0 = self.subtasks[0].predicate
                predicate1 = self.subtasks[1].predicate
            # NOTE: Always one seq subtask is Cons.ev in our spec lang
            # NOTE: Can't handle case where one is choose
            if self.subtasks[0].cons == Cons.ev:
                subval = self.subtasks[1].quantitative_semantics_dp(
                    sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[0][n-1] = -MAX_PRED_VAL
                    else:
                        retval[0][n-i-1] = max(retval[0][n-i],
                                               min(predicate0(sys_traj[n-i-1],
                                                              res_traj[n-i-1]),
                                                   subval[0][n-i]))
                        subval[0][n-i-1] = max(subval[0]
                                               [n-i], subval[0][n-i-1])
            if self.subtasks[1].cons == Cons.ev:
                subval = self.subtasks[0].quantitative_semantics_dp(
                    sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[1][0] = -MAX_PRED_VAL
                    else:
                        retval[1][i] = max(retval[1][i-1],
                                           min(predicate1(sys_traj[i],
                                                          res_traj[i]),
                                               subval[1][i-1]))
                        subval[1][i] = max(subval[1][i-1], subval[1][i])
            return retval

        # choice
        if self.cons == Cons.choose:
            subval1 = self.subtasks[0].quantitative_semantics_dp(
                sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(
                sys_traj, res_traj)
            for i in range(n):
                retval[0][i] = max(subval1[0][i], subval2[0][i])
                retval[1][i] = max(subval1[1][i], subval2[1][i])
            return retval

        # both
        if self.cons == Cons.both:
            subval1 = self.subtasks[0].quantitative_semantics_dp(
                sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(
                sys_traj, res_traj)
            for i in range(n):
                retval[0][i] = min(subval1[0][i], subval2[0][i])
                retval[1][i] = min(subval1[1][i], subval2[1][i])
            return retval

    # returns a monitor: Monitor_Automaton
    # global_mode : Set to False to prevent global states from being created
    def get_monitor(self, state_dim, res_dim, local_reward_bound,
                    create_global_monitor=False):
        global_states = []

        # atomic eventually task
        if self.cons == Cons.ev:

            mpred = monitor_predicate(self.predicate, state_dim)
            if create_global_monitor and self.is_global_mode():
                # include state dim of # agents
                state_dim = state_dim*self.num_agents
                mpred = monitor_predicate(self.gl_predicate, state_dim)
                global_states = [0, 1]

            def pred_update(state, reg):
                return np.array([mpred(state, reg)[1]])

            def frew(state, reg):
                return reg[0]

            e00 = (0, true_predicate(local_reward_bound), id_update)
            e01 = (1, mpred, pred_update)
            e11 = (1, true_predicate(local_reward_bound), id_update)
            transitions = [[e00, e01], [e11]]
            rewards = [None, frew]

            return Monitor_Automaton(2, 1, state_dim + res_dim, np.array([0.0]), transitions,
                                     rewards, num_agents=self.num_agents, global_states=global_states,
                                     task_info={0: self.info, 1: None})

        # adding safety constraints
        elif self.cons == Cons.alw:
            # NOTE: Modify to allow alw (local/global) with mspec.
            #  Maybe decouple the alw register? But decoupling will complicate.

            # Check if subtasks involve global variables
            st1_global = create_global_monitor and self.subtasks[0].is_global_mode(
            )

            # construct monitor for sub-formula
            mon = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound,
                                               create_global_monitor=st1_global)

            # add local safety constraint
            mpred = monitor_predicate(self.predicate, state_dim)

            if create_global_monitor:
                # Set global states
                global_states = mon.global_states
                if self.is_global_mode():
                    # include state dim of # agents
                    state_dim = state_dim*self.num_agents
                    mpred_gl = monitor_predicate(self.gl_predicate, state_dim)

            n_states = mon.n_states
            n_registers = mon.n_registers + 1
            input_dim = mon.input_dim
            init_reg = np.concatenate(
                [mon.init_registers, np.array([local_reward_bound])])

            transitions = []
            for ve in mon.transitions:
                ve_new = []
                # To get the current monitor state
                curr_q_value = len(transitions)
                for (q, p, u) in ve:
                    if curr_q_value in global_states:  # Assuming in global state, the always predicate is also global
                        ve_new.append((q,
                                       project_predicate(
                                           p, 0, mon.n_registers),
                                       alw_update(u, mpred_gl)))
                    else:  # Assuming in local state, the always predicate is also local
                        ve_new.append((q,
                                       project_predicate(
                                           p, 0, mon.n_registers),
                                       alw_update(u, mpred)))
                transitions.append(ve_new)

            rewards = []
            for rew in mon.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(alw_reward(rew))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards, num_agents=self.num_agents, global_states=global_states)

        # sequence
        elif self.cons == Cons.seq:

            # Check if subtasks involve global variables
            st1_global = create_global_monitor and self.subtasks[0].is_global_mode(
            )
            st2_global = create_global_monitor and self.subtasks[1].is_global_mode(
            )

            # construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st1_global)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st2_global)

            # Set global states
            if create_global_monitor:
                global_states = mon1.global_states

            # construct monitor for sequence
            n_states = mon1.n_states + mon2.n_states
            n_registers = max(mon1.n_registers, mon2.n_registers + 1)
            input_dim = max(mon1.input_dim, mon2.input_dim)
            # NOTE: MON 1 local and MON 2 global so diff input dimm
            # Fix: Make local predicate depend on first n dim of state
            # Keep ordering when not in global state
            init_reg = np.zeros(n_registers)
            init_reg[:mon1.n_registers] = mon1.init_registers

            # Same monitor states as Mon 1
            task_info = mon1.task_info

            transitions = []
            for qu in range(mon1.n_states):
                ve = mon1.transitions[qu]
                ve_new = []

                # Delta1
                for (qv, p, u) in ve:
                    ve_new.append((qv,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))

                # Delta1->2
                # only if qu is a final state in mon1
                # NOTE: What if final state is global and mon2 state is local?
                if mon1.rewards[qu] is not None:
                    for (qv, p, u) in mon2.transitions[0]:
                        q2 = qv + mon1.n_states
                        p2 = rew_pred(
                            p, mon1.rewards[qu], mon2.init_registers, 0, mon1.n_registers)
                        u2 = seq_update(n_registers, mon1.n_registers, mon2.n_registers,
                                        mon2.init_registers, mon1.rewards[qu], u)
                        ve_new.append((q2, p2, u2))
                        # Update info also for linking state
                        if task_info[qu] is None:  # Not updated yet
                            # Get info of first task of Mon2
                            task_info[qu] = mon2.task_info[0]

                    if create_global_monitor and 0 in mon2.global_states:
                        # If mon2 0 state is global (predicate p to 1 is global),
                        # Delta1->2 also global
                        global_states.append(qu)

                transitions.append(ve_new)

            for _ti, ve in enumerate(mon2.transitions):
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states,
                                   project_predicate(p, 0, mon2.n_registers),
                                   project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)
                # Update info for all other mon2 states
                task_info[len(transitions)-1] = mon2.task_info[_ti]
                if create_global_monitor and _ti in mon2.global_states:
                    # len(transitions)-1 is the current state in the new monitor
                    global_states.append(len(transitions)-1)

            rewards = [None]*mon1.n_states
            for rew in mon2.rewards:
                if rew is not None:
                    rewards.append(seq_reward(rew, mon2.n_registers))
                else:
                    rewards.append(None)

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards, self.num_agents, global_states=global_states,
                                     task_info=task_info)

        # choice
        elif self.cons == Cons.choose:
            # Need to check Synchronization State here

            # Check if subtasks involve global variables
            st1_global = create_global_monitor and self.subtasks[0].is_global_mode(
            )
            st2_global = create_global_monitor and self.subtasks[1].is_global_mode(
            )

            # adding Synchronization state if mixed
            if st1_global != st2_global:
                # if both states are global, no change
                # If mixed objective, add synchronization state to global task
                global_st_index = 1*int(st2_global)
                # Local predicate to provide synch state

                def true_pred(sys_state, reg_state):
                    return True
                new_subtask = seq(
                    ev(true_pred), self.subtasks[global_st_index], num_agents=self.num_agents)
                self.subtasks[global_st_index] = new_subtask

            # construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st1_global)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st2_global)

            # combine
            # initial state is merged, state numbers of first monitor do not change
            n_states = mon1.n_states + mon2.n_states - 1
            n_registers = mon1.n_registers + mon2.n_registers
            input_dim = max(mon1.input_dim, mon2.input_dim)
            global_states = mon1.global_states
            init_reg = np.concatenate(
                [mon1.init_registers, mon2.init_registers])

            transitions = []

            # Delta0
            # us[0] stores loop update for the first monitor, us[1] for second monitor
            us = []
            # Set of transitions from initial state: None is used as a placeholder for self loop
            trans_init = [None]
            for (q1, p1, u1) in mon1.transitions[0]:
                if q1 == 0:
                    us.append(u1)
                else:
                    trans_init.append((q1,
                                       project_predicate(
                                           p1, 0, mon1.n_registers),
                                       project_update(u1, 0, mon1.n_registers, clean=True)))

            for (q2, p2, u2) in mon2.transitions[0]:
                if q2 == 0:
                    us.append(u2)
                else:
                    trans_init.append((q2 + mon1.n_states - 1,
                                       project_predicate(
                                           p2, mon1.n_registers, n_registers),
                                       project_update(u2, mon1.n_registers, n_registers,
                                                      clean=True)))

            def loop_update(state, reg):
                return np.concatenate([us[0](state, reg[0:mon1.n_registers]),
                                       us[1](state, reg[mon1.n_registers:n_registers])])
            trans_init[0] = (0, true_predicate(
                local_reward_bound), loop_update)
            transitions.append(trans_init)

            # Delta1: Add all transitions in monitor 1
            for ve in mon1.transitions[1:]:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2: Add all transitions in monitor 2 and monitor 2 global states
            for _ti, ve in enumerate(mon2.transitions[1:]):
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states - 1,
                                   project_predicate(
                                       p, mon1.n_registers, n_registers),
                                   project_update(u, mon1.n_registers, n_registers)))
                transitions.append(ve_new)
                if create_global_monitor and _ti+1 in mon2.global_states:
                    # _ti+1 is state id in mon2 of current transition
                    # len(transitions)-1 is the current state in the new monitor
                    global_states.append(len(transitions)-1)

            rewards = [None]
            for rew in mon1.rewards[1:]:
                if rew is not None:
                    rewards.append(project_reward(rew, 0, mon1.n_registers))
                else:
                    rewards.append(None)

            for rew in mon2.rewards[1:]:
                if rew is not None:
                    rewards.append(project_reward(
                        rew, mon1.n_registers, n_registers))
                else:
                    rewards.append(None)

            # TODO: fill task_info appropriately
            task_info = {i: None for i in range(n_states)}

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards, self.num_agents, global_states=global_states,
                                     task_info=task_info)

        # both (or AND over specs). Used in centralized SPECTRL
        elif self.cons == Cons.both:

            # Check if subtasks involve global variables
            st1_global = create_global_monitor and self.subtasks[0].is_global_mode(
            )
            st2_global = create_global_monitor and self.subtasks[1].is_global_mode(
            )

            # adding synch state if mixed
            if st1_global != st2_global:
                # if both states are global, no change
                # If mixed objective, add synchronization state to global task
                global_st_index = 1*int(st2_global)
                # Local predicate to provide synch state

                def true_pred(sys_state, reg_state):
                    return True
                new_subtask = seq(
                    ev(true_pred), self.subtasks[global_st_index], num_agents=self.num_agents)
                self.subtasks[global_st_index] = new_subtask

            # construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st1_global)
            mon2 = self.subtasks[1].get_monitor(state_dim, res_dim, local_reward_bound,
                                                create_global_monitor=st2_global)

            # combine
            # states are multiplied since each monitor runs in parallel
            n_states = mon1.n_states * mon2.n_states
            n_registers = mon1.n_registers + mon2.n_registers
            input_dim = max(mon1.input_dim, mon2.input_dim)
            global_states = []  # Fill it up in transition phase
            init_reg = np.concatenate(
                [mon1.init_registers, mon2.init_registers])

            # None is used as a placeholder for each transition
            transitions = [None for i in range(n_states)]

            # Not merging initial state, no special case
            # Do all transitions

            # Delta1: Add all transitions in monitor 1
            for _ti, ve in enumerate(mon1.transitions):
                # _ti is state id in mon1 of current transition
                for j in range(mon2.n_states):  # Careful that right states being mapped
                    ve_new = []
                    for (q, p, u) in ve:
                        ve_new.append((j*mon1.n_states + q,
                                       project_predicate(
                                           p, 0, mon1.n_registers),
                                       project_update(u, 0, mon1.n_registers)))
                    transitions[j*mon1.n_states + _ti] = ve_new

            # Delta2: Add all transitions in monitor 2
            for _ti, ve in enumerate(mon2.transitions):
                for k in range(mon1.n_states):  # Careful that right states being mapped
                    ve_new = []
                    for (q, p, u) in ve:
                        ve_new.append((q*mon1.n_states + k,
                                       project_predicate(
                                           p, mon1.n_registers, n_registers),
                                       project_update(u, mon1.n_registers, n_registers)))
                    # combine w/ existing transitions
                    transitions[_ti*mon1.n_states + k] += ve_new

            # Merge all self-loops into one transition for a fair comparison b/w centralized and dist
            if self.merge_self_loop:
                # Iterate over transitions merging all self loops into one update+predicate
                for i, ve in enumerate(transitions):
                    ind_to_remove = []
                    for j, (q, p, u) in enumerate(ve):
                        if q == i:
                            ind_to_remove.append(j)
                    if len(ind_to_remove) > 0:
                        merge_ve = [ve[j] for j in ind_to_remove]
                        opt_pred = optimistic_predicate_merge(
                            *[t[1] for t in merge_ve])
                        opt_update = optimistic_update_merge(
                            *[t[2] for t in merge_ve])

                        # Merged transition
                        new_ve = [(i, opt_pred, opt_update)]
                        new_ve += [t for j,
                                   t in enumerate(ve) if j not in ind_to_remove]
                        transitions[i] = new_ve

            rewards = []
            for rew2 in mon2.rewards:
                for rew in mon1.rewards:
                    # Only create reward state for those with BOTH rewards existing.
                    # min() to ensure AND
                    if rew is not None and rew2 is not None:
                        rewards.append(min_reward(project_reward(rew, 0, mon1.n_registers),
                                                  project_reward(rew2, mon1.n_registers, n_registers)))
                    else:
                        rewards.append(None)

            # TODO: fill task_info appropriately
            task_info = {i: None for i in range(n_states)}

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards, self.num_agents, global_states=global_states,
                                     task_info=task_info)

        # conditional
        else:

            # Construct monitors for sub formulas
            mon1 = self.subtasks[0].get_monitor(
                state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(
                state_dim, res_dim, local_reward_bound)

            b = monitor_predicate(self.predicate, state_dim)
            notb = neg_pred(b)

            # Combine monitors
            n_states = mon1.n_states + mon2.n_states + 1
            n_registers = max(mon1.n_registers, mon2.n_registers)
            input_dim = mon1.input_dim
            init_reg = np.zeros(n_registers)

            transitions = []

            # Delta0
            trans_init = []

            for (q, p, u) in mon1.transitions[0]:
                trans_init.append((q+1,
                                   conj_pred(p, b, mon1.init_registers),
                                   init_update(u, mon1.init_registers)))
            for (q, p, u) in mon2.transitions[0]:
                trans_init.append((q + mon1.n_states + 1,
                                   conj_pred(p, notb, mon2.init_registers),
                                   init_update(u, mon2.init_registers)))
            transitions.append(trans_init)

            # Delta1

            for ve in mon1.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q+1,
                                   project_predicate(p, 0, mon1.n_registers),
                                   project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2

            for ve in mon2.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states + 1,
                                   project_predicate(p, 0, mon2.n_registers),
                                   project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)

            rewards = [None]

            for rew in mon1.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(project_reward(rew, 0, mon1.n_registers))

            for rew in mon2.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(project_reward(rew, 0, mon2.n_registers))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

# ==================================================================================================
# API for building Task Specifications


# Logical OR
# p1, p2 : np.array([state_dim]) * np.array([res_dim]) -> Float
def lor(p1, p2):
    def p(sys_state, res_state):
        return max(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


# Logical AND
# p1, p2 : np.array([state_dim]) * np.array([res_dim]) -> Float
def land(p1, p2):
    def p(sys_state, res_state):
        return min(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


# Atomic eventually task
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
def ev(p, num_agents=1, gl_predicate=None, info=None):
    return TaskSpec(Cons.ev, p, [], num_agents, gl_predicate, info=info)


# Safety constraints
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
# phi : TaskSpec
def alw(p, phi, gl_predicate=None):
    return TaskSpec(Cons.alw, p, [phi], gl_predicate=gl_predicate)


# Sequence of tasks
# phi1, phi2 : TaskSpec
def seq(phi1, phi2, num_agents=1):
    return TaskSpec(Cons.seq, None, [phi1, phi2], num_agents=num_agents)


# Choice of tasks (OR)
# phi1, phi2 : TaskSpec
def choose(phi1, phi2):
    return TaskSpec(Cons.choose, None, [phi1, phi2])


# Both of tasks (AND)
# phi1, phi2 : TaskSpec
# merge_self_loop: True since we ONLY use this composition for centralized monitor
def both(phi1, phi2):
    return TaskSpec(Cons.both, None, [phi1, phi2], merge_self_loop=True)


# Conditional
# p : np.array([state_dim]) * np.array([res_dim]) -> Float
# phi1, phi2 : TaskSpec
def ite(p, phi1, phi2):
    return TaskSpec(Cons.ite, p, [phi1, phi2])
