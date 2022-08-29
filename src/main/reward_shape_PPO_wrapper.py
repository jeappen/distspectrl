from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from typing import Dict
from builtins import max
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

from gym_waysharing.nav_env import NavEnv, NavEnv3D

import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def env_creator(config):
    spec_id = config['spec_id'] if 'spec_id' in config else 1
    err_arg = config['err_arg'] if 'err_arg' in config else 0
    max_mode_arg = config['max_mode'] if 'max_mode' in config else False
    num_agents_arg = config['num_agents'] if 'num_agents' in config else 3
    horizon_arg = config['horizon'] if 'horizon' in config else 400
    DS_mode_arg = config['distributed_mode'] if 'distributed_mode' in config else False
    limit_pred_val_arg = config['limit_pred_val'] if 'limit_pred_val' in config else False

    print('current spec_id {}, err arg{}'.format(spec_id, err_arg))
    base_env = NavEnv(n_agents=num_agents_arg, time_horizon=horizon_arg)
    from src.env_wrappers.MA_learning import MA_ProductMDP
    num_agents = base_env.n_agents
    state_dim = base_env.nx_system
    x_dim = 2  # For 2-D state

    # Reward scaling config
    # Used to scale predicate values and set lower bounds for the shaped reward
    lb_scale = config['reward_scale']['lb_scale'] if 'reward_scale' in config else 5
    # Used set lower bounds for the shaped reward
    lb_scale_factor = config['reward_scale']['lb_scale_factor'] if 'reward_scale' in config else 2.0
    # Used to scale predicate values
    predicate_scaling = config['reward_scale']['predicate_scaling'] if 'reward_scale' in config else 1.0

    # Multispec options
    multispec_ids = config['multispec'] if 'multispec' in config else []

    # logN scaling config
    log_n_k = config['log_n_scaling']['log_n_k'] if 'log_n_scaling' in config else 3.0
    log_n_factor = config['log_n_scaling']['log_n_factor'] if 'log_n_scaling' in config else 2

    from src.main.MA_spec_compiler import ev, seq, choose, both

    # Define the specification
    # NOTE: Local spec should only work on the first S_a dimensions
    #       Sorted with current agent first
    # 1. Relevant atomic local predicates:
    # a. Reach predicate
    #    goal: np.array(1), err: float
    # wrt point if reference frame is at dim 2,3
    def reach(goal, err, wrt_point=True, limit_pred_val=True, scaling_pred=2):
        def predicate(sys_state, res_state):
            x = sys_state[0]
            y = sys_state[1]
            if wrt_point:
                x -= sys_state[2]
                y -= sys_state[3]
            op_value = (min([x - goal[0],
                             goal[0] - x,
                             y - goal[1],
                             goal[1] - y]) + err)
            return max(-lb_scale * err, op_value / scaling_pred) if limit_pred_val else op_value

        return predicate

    # b. Avoid predicate
    # Not used in DistSPECTRL
    #    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
    def avoid(obstacle):
        def predicate(sys_state, res_state):
            return max([obstacle[0] - sys_state[0],
                        obstacle[1] - sys_state[1],
                        sys_state[0] - obstacle[2],
                        sys_state[1] - obstacle[3]])

        return predicate

    # Goals and obstacles
    gtop = np.array([5.0, 10.0])
    gbot = np.array([5.0, 0.0])
    gright = np.array([10.0, 0.0])
    gcorner = np.array([10.0, 10.0])
    gcorner2 = np.array([0.0, 10.0])
    origin = np.array([0.0, 0.0])
    obs = np.array([4.0, 4.0, 6.0, 6.0])
    gbot_closer = np.array([3.0, 0.0])
    gtop_origin_closer = np.array([0, 3])
    gtop_closer = np.array([5.0, 3.0])
    gcenter = np.array([5.0, 5.0])
    err = 1.0 if err_arg <= 0 \
        else err_arg
    wrt_point = False  # When using full dim system, to centre dims 1,2 on dims 3,4

    # Local Specifications
    # Made for a MA system w num_agents>1
    lambda_spec_goal_lo = lambda x: ev(reach(x, err, wrt_point=wrt_point,
                                             limit_pred_val=limit_pred_val_arg, scaling_pred=predicate_scaling),
                                       #             gl_predicate=reach_gl(x, err, num_agents,max_mode_arg),
                                       info=x,
                                       num_agents=num_agents)  # Simple local spec in global setting

    # Global Specifications
    # These work on all dimensions. State input is ordered by agent index
    def reach_gl(goal, err, num_agents, max_mode=False, limit_pred_val=True, scaling_pred=2, reference_goal_arg=None, test_arg=0):
        # reference_goal_arg: If given, enable reach_gl to have different goals
        # Repeat reach for global
        def predicate(sys_state, res_state):
            # Every predicate has a global approximation option
            check_if_approx = ((sys_state.shape[0] // num_agents) != state_dim)
            if check_if_approx:
                scaling = sys_state.shape[0] // state_dim
                # Now act on \phi(S) = S_A ^ k
            else:
                scaling = num_agents
            sys_state_extracted = np.hstack([sys_state[si * state_dim:si * state_dim + x_dim]
                                             for si in range(scaling)])
            if reference_goal_arg is None:
                reference_goal = np.tile(goal, scaling)
            else:
                reference_goal = reference_goal_arg
            if max_mode:
                op_value = -np.linalg.norm(sys_state_extracted - reference_goal,
                                           ord=np.Inf) + err
            else:
                op_value = -np.linalg.norm(sys_state_extracted - reference_goal,
                                           ord=1) / scaling + err
            return max(-lb_scale * err, op_value / scaling_pred) if limit_pred_val else op_value

        return predicate

    lambda_spec_goal_gl = lambda x: ev(reach(x, err, wrt_point=wrt_point,
                                             limit_pred_val=limit_pred_val_arg, scaling_pred=predicate_scaling),
                                       gl_predicate=reach_gl(x, err, num_agents, max_mode_arg,
                                                             limit_pred_val=limit_pred_val_arg,
                                                             scaling_pred=predicate_scaling),
                                       info=x,
                                       num_agents=num_agents)  # Simple global spec

    # To specify different goals for different agents
    lambda_spec_multi_goal_global = lambda x, goals: ev(reach(x, err, wrt_point=wrt_point,
                                                              limit_pred_val=limit_pred_val_arg, scaling_pred=predicate_scaling),
                                                        gl_predicate=reach_gl(x, err, num_agents, max_mode_arg,
                                                                              limit_pred_val=limit_pred_val_arg,
                                                                              scaling_pred=predicate_scaling,
                                                                              reference_goal_arg=goals),
                                                        info=x,
                                                        num_agents=num_agents)  # Simple global spec

    # Test Global Specifications
    ''' To test if centralized reach can scale
        reach_lo_asgl() to check reach goal for i'th agent alone
        gives a predicate AND(ev(reach_i(s,g))), reach_gl is ev(AND(reach_i(s,g)))
    '''

    def reach_i(goal, err, num_agents, i=0, max_mode=False):
        def predicate(sys_state, res_state):

            # print(i,state_dim,x_dim,sys_state,'DEBUG')
            # assert(i*state_dim+x_dim <= len(sys_state))
            if i * state_dim + x_dim <= len(sys_state):
                sys_state_extracted = sys_state[i *
                                                state_dim:i * state_dim + x_dim]
            else:
                # To get around the init env with one agent but num_agents 2?
                # i=0
                sys_state_extracted = sys_state[:x_dim]

            if max_mode:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           goal, ord=np.Inf) + err
            else:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           goal, ord=1) + err

            return op_value

        return predicate

    # Centralized spec for reach_i
    lambda_spec_goal_gl_centralized = lambda x: ev(reach_gl(x, err, num_agents, max_mode_arg,
                                                            limit_pred_val=limit_pred_val_arg,
                                                            scaling_pred=predicate_scaling),
                                                   info=x)
    lambda_spec_goal_reach_i = lambda x, i: ev(
        reach_i(x, err, num_agents, i, max_mode_arg))

    def reach_lo_asgl(goal):
        """Local Spec lambda_spec_goal_reach_i on each agent combined to form a centralized version"""
        op = both(lambda_spec_goal_reach_i(goal, 0),
                  lambda_spec_goal_reach_i(goal, 1))
        for i in range(2, num_agents):
            op = both(op, lambda_spec_goal_reach_i(goal, i))
        return op

    # To keep one of seq tasks as ev()
    def reach_lo_asgl_w_seq_task(goal, task2):
        op = both(seq(lambda_spec_goal_reach_i(goal, 0), task2),
                  seq(lambda_spec_goal_reach_i(goal, 1), task2))
        for i in range(2, num_agents):
            op = both(op, seq(lambda_spec_goal_reach_i(goal, i), task2))
        return op

    def combine_using_binary_op(i, binary_op, *args):
        # Used to make a series of reach_i sequential in postfix fashion
        op = binary_op(args[0](i), args[1](i))
        for a in args[2:]:
            op = binary_op(op, a(i))
        return op

    # Seq of several reach_i as global spec
    # binary_op : seq or choose
    # args: lambda i
    #            i : agent index,
    #       output : predicate
    def concat_local_bin_op_asgl(binary_op, *args):
        # TODO: use global_end_spec if specified (global_end_spec=None)
        # Combine a local spec as global for agent i. eg: reach_i(P) ; reach_i(Q)
        get_seq_task = lambda i: combine_using_binary_op(i, binary_op, *args)
        op = both(get_seq_task(0), get_seq_task(1))
        for i in range(2, num_agents):
            op = both(op, get_seq_task(i))
        return op

    # Gets lambda fn of agent index i, reach i and given goal
    def get_lambda_reach_i(goal):
        return lambda i: lambda_spec_goal_reach_i(goal, i)

    phi_0_centralized = lambda_spec_goal_gl_centralized(origin)
    phi_1_centralized = seq(lambda_spec_goal_gl_centralized(
        gbot), phi_0_centralized, num_agents=num_agents)  # \phi_1
    phi_2_centralized = seq(phi_1_centralized, lambda_spec_goal_gl_centralized(
        gbot_closer), num_agents=num_agents)  # \phi_2
    phi_3a_centralized = seq(reach_lo_asgl(
        gbot), phi_0_centralized, num_agents=num_agents)  # helper to build \phi_3
    phi_3_centralized = seq(phi_3a_centralized, lambda_spec_goal_gl_centralized(
        gbot_closer), num_agents=num_agents)  # \phi_3
    # HACK: MODIFIED to really capture local spec
    # Local part of mspec4
    local_choose_reach_seq_1 = concat_local_bin_op_asgl(seq,
                                                        lambda i: combine_using_binary_op(i, choose,
                                                                                          get_lambda_reach_i(
                                                                                              gbot_closer),
                                                                                          get_lambda_reach_i(gtop)),
                                                        get_lambda_reach_i(gbot))
    # Combined spec
    phi_4_centralized = seq(seq(local_choose_reach_seq_1, phi_0_centralized),
                            lambda_spec_goal_gl_centralized(gbot_closer))  # \phi_4

    if not DS_mode_arg:
        spec = [phi_0_centralized, phi_1_centralized, phi_2_centralized,
                phi_3_centralized, phi_4_centralized, phi_3a_centralized][spec_id]

    # MULTISPEC TOOLS (not in ecml release)
    # Building blocks to allow different specs for different agents

    # Example multi_spec, lo1,gl and lo2,gl for 2 agents
    # Same 'structure' different semantics for local specs
    # ASSUMPTION: Global objectives remain the same i.e. reach_gl(X) has X same for each spec in multispec.
    dummy_spec0 = lambda_spec_goal_gl(origin)
    sample_multispec_medium = [seq(lambda_spec_goal_lo(gbot), dummy_spec0, num_agents=num_agents),
                               seq(lambda_spec_goal_lo(gtop_origin_closer), dummy_spec0, num_agents=num_agents)]
    sample_multispec_easy = [lambda_spec_goal_lo(gbot),
                             lambda_spec_goal_lo(origin)]

    sample_multispec_dummy = [lambda_spec_goal_gl(origin),
                              lambda_spec_goal_gl(origin)]

    sample_multispec = sample_multispec_medium

    sample_multispec_map = lambda agent_id: int(
        agent_id[-1]) % len(sample_multispec)  # handles agent-0, agent-1
    # for when multispec is part of the config
    multispec_map = lambda agent_id: int(agent_id[-1]) % len(multispec_ids)

    if DS_mode_arg:
        # Global Specs
        phi_0 = lambda_spec_goal_gl(origin)
        phi_1 = seq(lambda_spec_goal_gl(gbot), phi_0,
                    num_agents=num_agents)  # \phi_1
        phi_2 = seq(phi_1, lambda_spec_goal_gl(gbot_closer),
                    num_agents=num_agents)  # \phi_2
        # Mixed spec
        mspec0 = seq(lambda_spec_goal_lo(gbot), phi_0, num_agents=num_agents)
        phi_3 = seq(mspec0, lambda_spec_goal_gl(gbot_closer),
                    num_agents=num_agents)  # \phi_3
        phi_4 = seq(choose(lambda_spec_goal_lo(gbot_closer),
                           lambda_spec_goal_lo(gtop)), phi_3)  # \phi_4

        dist_spec_list = [phi_0, phi_1, phi_2, phi_3, phi_4]
        spec = dist_spec_list[spec_id]
        if len(multispec_ids) == 0:
            multispec_map = sample_multispec_map
            multispec = None
        else:
            multispec = [dist_spec_list[ms_id] for ms_id in multispec_ids]

    if 'inline_reward' in config and config['inline_reward']:
        # To calculate reward within the env for use with ANY MA algorithm
        if 'spectrl_logn' in config and config['spectrl_logn']:
            # To use logN scaling method with Inline Rewards
            from src.env_wrappers.MA_learning_logNscaling import InlineMAProductMDPLogNScaling
            # Is a distributed spec, pick from dist_spec_pool via DS_mode_arg
            final_env = InlineMAProductMDPLogNScaling(base_env, base_env.action_space.shape[0], spec,
                                                      0.0, lb_scale_factor * lb_scale * err,
                                                      log_n_k=min(
                                                          log_n_k, num_agents),
                                                      log_n_scaling_rw_ub=lb_scale * err,
                                                      log_n_factor=log_n_factor)

        elif 'no_monitor' in config and config['no_monitor']:
            # To allow baseline without monitor state
            from src.env_wrappers.MA_learning import MA_ProductMDP_withRandTaskMonitor
            final_env = MA_ProductMDP_withRandTaskMonitor(base_env, base_env.action_space.shape[0], spec, 0.0,
                                                          lb_scale_factor * lb_scale * err)
        else:
            from src.env_wrappers.MA_learning import InlineMA_ProductMDP
            # Is a distributed spec, pick from dist_spec_pool via DS_mode_arg
            final_env = InlineMA_ProductMDP(base_env, base_env.action_space.shape[0], spec,
                                            0.0, lb_scale_factor * lb_scale * err,
                                            multi_specs=multispec, multi_spec_map=multispec_map)
    elif 'spectrl_single_agent' in config and config['spectrl_single_agent']:
        # To allow training with centralized Single agent methods
        from src.env_wrappers.MA_learning import MAEnvtoSA, SA_ProductMDP
        base_sa_env = MAEnvtoSA(base_env, spec, zero_reward=True)
        final_env = SA_ProductMDP(
            base_sa_env, base_sa_env.action_dim, spec, 0.0, lb_scale_factor * lb_scale * err)
    elif 'tltl_single_agent' in config and config['tltl_single_agent']:
        # To allow training with centralized Single agent methods
        from src.env_wrappers.MA_learning import MAEnvtoSA
        final_env = MAEnvtoSA(base_env, spec)
    else:
        # Is a distributed spec, pick from dist_spec_pool via DS_mode_arg
        final_env = MA_ProductMDP(
            base_env, base_env.action_space.shape[0], spec, 0.0, lb_scale_factor * lb_scale * err)

    return final_env


def get_important_members(single_env):
    # Get env functions required for postprocessing
    env_config = {'spec': single_env.spec}
    from src.env_wrappers.MA_learning import MAEnvtoSA, SA_ProductMDP, reshape_state_for_spec
    if not isinstance(single_env, MAEnvtoSA) and not isinstance(single_env, SA_ProductMDP):
        env_config['cum_reward'] = single_env.cum_reward
        env_config['reshape_state_for_spec'] = reshape_state_for_spec
        env_config['global_mode'] = single_env.global_mode
    return env_config


def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """
    sample_size = 5

    for i in range(sample_size):
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    metrics = summarize_episodes(episodes)
    output_metric = dict(
        episode_reward_max=metrics['episode_reward_max'],
        episode_reward_min=metrics['episode_reward_min'],
        episode_reward_mean=metrics['episode_reward_mean'],
        episode_len_mean=metrics['episode_len_mean'],
        custom_metrics=metrics['custom_metrics'])
    return output_metric


custom_metric_name = 'max_monitor_state'
custom_metric_name2 = 'max_depth'
# To check for satisfaction of spec wrt task monitor
custom_metric_name3 = 'final_state_reached'
# To check for satisfaction of spec wrt task monitor
custom_metric_name4 = 'stage_reached'


class CustomCallbacks(DefaultCallbacks):
    # Uses monitor state to determine if Spec is satisfied
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {}  started.".format(
        #     episode.episode_id))
        episode.user_data[custom_metric_name] = []
        episode.hist_data[custom_metric_name] = []
        if 'spectrl_logn' in worker.policy_config['env_config'] and worker.policy_config['env_config']['spectrl_logn']:
            # Include stage and extract monitor_state correctly
            episode.user_data[custom_metric_name4] = []
            episode.hist_data[custom_metric_name4] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):

        log_n_scale_mode = custom_metric_name4 in episode.user_data
        if log_n_scale_mode:
            # Last dim is actually current stage
            current_stages = []
            for aid in episode._agent_to_index:
                current_stage = episode.last_raw_obs_for(aid)[-1]
                current_stages.append(current_stage)
            episode.user_data[custom_metric_name4].append(current_stages)

        monitor_states = []
        for aid in episode._agent_to_index:
            # iterate over agents
            monitor_state = episode.last_raw_obs_for(
                aid)[-2] if log_n_scale_mode else episode.last_raw_obs_for(aid)[-1]
            monitor_states.append(monitor_state)
        episode.user_data[custom_metric_name].append(monitor_states)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        final_monitor_state = np.min([episode.last_info_for(
            aid)['monitor_state'] for aid in episode._agent_to_index])
        # print("episode {} ended with length {} and min final_monitor_state "
        #       " {}".format(episode.episode_id, episode.length,
        #                          final_monitor_state))
        episode.custom_metrics[custom_metric_name] = final_monitor_state
        log_n_scale_mode = custom_metric_name4 in episode.user_data
        if log_n_scale_mode:
            # Last dim is actually current stage
            current_stage = np.min(episode.user_data[custom_metric_name4][-1])
            episode.custom_metrics[custom_metric_name4] = current_stage

        final_depth = worker.policy_config['env_config']['spec'].depths[int(
            final_monitor_state)]
        episode.custom_metrics[custom_metric_name2] = final_depth
        spec_satisfied = worker.policy_config['env_config']['spec'].monitor.rewards[
            int(final_monitor_state)] is not None
        episode.custom_metrics[custom_metric_name3] = int(spec_satisfied)

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


def env3D_creator(config):
    # Creates specs for NavEnv3D env
    spec_id = config['spec_id'] if 'spec_id' in config else 1
    err_arg = config['err_arg'] if 'err_arg' in config else 0
    max_mode_arg = config['max_mode'] if 'max_mode' in config else False
    # num_agents_arg = config['num_agents'] if 'num_agents' in config else 3
    # horizon_arg = config['horizon'] if 'horizon' in config else 400
    limit_pred_val_arg = config['limit_pred_val'] if 'limit_pred_val' in config else False

    print('NavEnv3D current spec_id {}, err arg{}'.format(spec_id, err_arg))

    base_env = NavEnv3D()
    from src.env_wrappers.MA_learning import MA_ProductMDP
    num_agents = base_env.n_agents
    state_dim = base_env.nx_system
    x_dim = 3  # For 2-D state

    # Reward scaling config
    # Used to scale predicate values and set lower bounds for the shaped reward
    lb_scale = config['reward_scale']['lb_scale'] if 'reward_scale' in config else 5
    # Used set lower bounds for the shaped reward
    lb_scale_factor = config['reward_scale']['lb_scale_factor'] if 'reward_scale' in config else 2.0
    # Used to scale predicate values
    predicate_scaling = config['reward_scale']['predicate_scaling'] if 'reward_scale' in config else 1.0

    log_n_k = config['log_n_scaling']['log_n_k'] if 'log_n_scaling' in config else 3.0

    from src.main.MA_spec_compiler import ev, seq, choose, alw, both

    # Define the specification
    # NOTE: Local spec should only work on the first S_a dimensions
    #       Sorted with current agent first
    # 1. Relevant atomic local predicates:
    # a. Reach predicate
    #    goal: np.array(1), err: float
    # wrt point if reference frame is at dim 2,3
    def reach(goal, err, wrt_point=True):
        def predicate(sys_state, res_state):
            x = sys_state[0]
            y = sys_state[1]
            z = sys_state[2]
            if wrt_point:
                x -= sys_state[3]
                y -= sys_state[4]
                z -= sys_state[5]
            return (min([x - goal[0],
                         goal[0] - x,
                         y - goal[1],
                         goal[1] - y,
                         z - goal[2],
                         goal[2] - z]) + err)

        return predicate

    # b. Avoid predicate
    # Not used in DistSPECTRL
    #    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
    def avoid(obstacle):
        def predicate(sys_state, res_state):
            return max([obstacle[0] - sys_state[0],
                        obstacle[1] - sys_state[1],
                        sys_state[0] - obstacle[2],
                        sys_state[1] - obstacle[3]])

        return predicate

    # Goals and obstacles
    gbot = np.array([5.0, 0.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    gbot_closer = np.array([3.0, 0.0, -3.0])
    err = 1.0 if err_arg <= 0 \
        else err_arg

    lb_scale = 2
    wrt_point = False

    # Global Specifications
    # These work on all dimensions. State input is ordered by agent index
    def reach_gl(goal, err, num_agents, max_mode=False, limit_pred_val=True, scaling_pred=2, reference_goal_arg=None, test_arg=0):
        # Repeat reach for global
        def predicate(sys_state, res_state):
            # Every predicate has a global approximation option
            check_if_approx = (
                sys_state.shape[0] // num_agents - state_dim + x_dim) != goal.shape[0]
            if check_if_approx:
                scaling = sys_state.shape[0] // (goal.shape[0] +
                                                 state_dim - x_dim)
                sys_state_extracted = np.hstack([sys_state[si * state_dim:si * state_dim + x_dim]
                                                 for si in range(scaling)])
                # Now act on \phi(S) = S_A ^ k
            else:
                scaling = num_agents
                sys_state_extracted = np.hstack([sys_state[si * state_dim:si * state_dim + x_dim]
                                                 for si in range(num_agents)])
            if max_mode:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           np.tile(goal, scaling), ord=np.Inf) + err
            else:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           np.tile(goal, scaling), ord=1) / scaling + err
            return max(-lb_scale * err, op_value / scaling_pred) if limit_pred_val else op_value

        return predicate

    lambda_spec_goal_gl = lambda x: ev(reach(x, err, wrt_point=wrt_point),
                                       gl_predicate=reach_gl(
                                           x, err, num_agents, max_mode_arg),
                                       num_agents=num_agents)  # Simple global spec

    # Test Global Specifications
    ''' To test if centralized reach can scale
        reach_lo_asgl() to check reach goal for i'th agent alone
        reach_lo yields predicate AND(ev(reach_i(s,g))), reach_gl is ev(AND(reach_i(s,g)))
    '''

    def reach_i(goal, err, num_agents, i=0, max_mode=False):
        def predicate(sys_state, res_state):

            if i * state_dim + x_dim <= len(sys_state):
                sys_state_extracted = sys_state[i *
                                                state_dim:i * state_dim + x_dim]
            else:
                # To get around the init env with one agent but num_agents 2?
                # i=0
                sys_state_extracted = sys_state[:x_dim]

            if max_mode:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           goal, ord=np.Inf) + err
            else:
                op_value = -np.linalg.norm(sys_state_extracted -
                                           goal, ord=1) + err
            return op_value

        return predicate

    # Centralized spec for reach_i
    lambda_spec_goal_gl_centralized = lambda x: ev(reach_gl(x, err, num_agents, max_mode_arg,
                                                            limit_pred_val=limit_pred_val_arg,
                                                            scaling_pred=predicate_scaling),
                                                   info=x)
    lambda_spec_goal_reach_i = lambda x, i: ev(
        reach_i(x, err, num_agents, i, max_mode_arg))

    def reach_lo_asgl(goal):
        """Local Spec lambda_spec_goal_reach_i on each agent combined to form a centralized version"""
        op = both(lambda_spec_goal_reach_i(goal, 0),
                  lambda_spec_goal_reach_i(goal, 1))
        for i in range(2, num_agents):
            op = both(op, lambda_spec_goal_reach_i(goal, i))
        return op

    # Global Specs
    spec0 = lambda_spec_goal_gl(origin)
    mspec0 = seq(ev(reach(gbot, err)), spec0, num_agents=num_agents)
    phi_a = seq(mspec0, lambda_spec_goal_gl(gbot_closer),
                num_agents=num_agents)  # \phi_a

    spec0_centralized = lambda_spec_goal_gl_centralized(origin)
    mspec0_central = seq(reach_lo_asgl(
        gbot), spec0_centralized, num_agents=num_agents)

    phi_a_central = seq(mspec0_central, lambda_spec_goal_gl_centralized(
        gbot_closer), num_agents=num_agents)

    spec = [phi_a, phi_a_central][spec_id]

    multispec_map = None
    multispec = None

    if 'inline_reward' in config and config['inline_reward']:
        # To calculate reward within the env for use with ANY MA algorithm
        if 'spectrl_logn' in config and config['spectrl_logn']:
            # To use logN scaling method with Inline Rewards
            from src.env_wrappers.MA_learning_logNscaling import InlineMAProductMDPLogNScaling
            # Is a distributed spec, pick from dist_spec_pool via DS_mode_arg
            final_env = InlineMAProductMDPLogNScaling(base_env, base_env.action_space.shape[0], spec,
                                                      0.0, lb_scale_factor * lb_scale * err,
                                                      log_n_k=min(
                                                          log_n_k, num_agents),
                                                      log_n_scaling_rw_ub=lb_scale * err)
        else:
            from src.env_wrappers.MA_learning import InlineMA_ProductMDP
            # Is a distributed spec, pick from dist_spec_pool via DS_mode_arg
            final_env = InlineMA_ProductMDP(base_env, base_env.action_space.shape[0], spec,
                                            0.0, lb_scale_factor * lb_scale * err,
                                            multi_specs=multispec, multi_spec_map=multispec_map)
    elif 'no_monitor' in config and config['no_monitor']:
        # To allow baseline without monitor state
        from src.env_wrappers.MA_learning import MA_ProductMDP_withRandTaskMonitor
        final_env = MA_ProductMDP_withRandTaskMonitor(base_env, base_env.action_space.shape[0], spec, 0.0,
                                                      lb_scale * err)
    elif 'spectrl_single_agent' in config and config['spectrl_single_agent']:
        # To allow training with centralized Single agent methods
        from src.env_wrappers.MA_learning import MAEnvtoSA, SA_ProductMDP
        base_sa_env = MAEnvtoSA(base_env, spec, zero_reward=True)
        final_env = SA_ProductMDP(
            base_sa_env, base_sa_env.action_dim, spec, 0.0, lb_scale_factor * lb_scale * err)
    else:
        final_env = MA_ProductMDP(
            base_env, base_env.action_space.shape[0], spec, 0.0, lb_scale * err)

    return final_env
