from ray.tune.integration.wandb import WandbLoggerCallback
from calendar import c
import ray
import os
from ray import tune
from ray.rllib.agents.registry import get_agent_class
import argparse

from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument(
    '--exp_name', default=None,
    help='Name of the ray_results experiment directory where results are stored.')
parser.add_argument(
    '--env', default='navenv',
    help='Name of the environment to rollout. Can be cleanup or harvest.')
parser.add_argument(
    '--horizon', default=400, type=int,
    help='Time Horizon used for when env is done')
parser.add_argument(
    '--algorithm', default='PPO',
    help='Name of the rllib algorithm to use.')
parser.add_argument(
    '--num_agents', default=3, type=int,
    help='Number of agent policies')
parser.add_argument(
    '--train_batch_size', default=30000, type=int,
    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument(
    '--checkpoint_frequency', default=20, type=int,
    help='Number of steps before a checkpoint is saved.')
parser.add_argument(
    '--training_iterations', default=10000, type=int,
    help='Total number of steps to train for')
parser.add_argument(
    '--num_cpus', default=2, type=int,
    help='Number of available CPUs')
parser.add_argument(
    '--num_gpus', default=1, type=int,
    help='Number of available GPUs')
parser.add_argument(
    '--use_gpus_for_workers', default=False,
    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument(
    '--use_gpu_for_driver', default=False,
    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument(
    '--num_workers_per_device', default=2, type=int,
    help='Number of workers to place on a single device (CPU or GPU)')

parser.add_argument(
    '--spec_id', default=3, type=int,
    help='Specification used from set of specs')
parser.add_argument(
    '--wandb', default=False,
    help='Tracks experiments using wandb.')

FLAGS = parser.parse_args()

default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.00001,
    'entropy_coeff': .00176}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1, spec_id=0, horizon=400):
    no_monitor_mode = False
    centralized_ppo_mode = 'CPPO' in algorithm
    individual_model_mode = False
    eval_env_config = None
    custom_env = None
    if 'rmachine' in env:
        # Use reward machine based on Reward Machines for Cooperative Multi-Agent Reinforcement Learning, Cyrus Neary et al
        # import Env with reward machine
        from rm_cooperative_marl.src.experiments.ray_rm_env import rm_env_creator, CustomCallbacks
        from src.main.reward_shape_PPO_wrapper import custom_eval_function
        custom_env = lambda config: rm_env_creator(config)
        env_config = {}
        # Choose spec from list of specs in env_creator
        env_config['spec_id'] = spec_id
        # Set to None for no reward shaping
        env_config['reward_shape_mode'] = 'sparse'
        # For setting predicate bounds
        env_config['err_arg'] = 0
        # For setting maximum distance mode in predicate
        env_config['max_mode'] = True
        # For setting num_agents
        env_config['num_agents'] = num_agents
        # For setting time horizon
        env_config['horizon'] = horizon

        env_config['distributed_mode'] = True
        env_config['inline_reward'] = True

        # Set eval mode to not use probability threshold in labelling and do actual MA env
        eval_env_config = env_config.copy()
        eval_env_config['marl_rm_eval_mode'] = True
        eval_fn = custom_eval_function
        single_env = custom_env(env_config)

    elif 'nav' in env:
        # for the navigation environments
        from src.main.reward_shape_PPO_wrapper import get_important_members
        from src.main.reward_shape_PPO_wrapper import custom_eval_function, CustomCallbacks
        eval_fn = custom_eval_function
        if '3D' in env:
            # for 3D environment
            from src.main.reward_shape_PPO_wrapper import env3D_creator
            env_creator = env3D_creator
        else:
            # for 2D environment
            from src.main.reward_shape_PPO_wrapper import env_creator
        custom_env = lambda config: env_creator(config)
        env_config = {}
        # Choose spec from list of specs in env_creator
        env_config['spec_id'] = spec_id
        # Set to None for no reward shaping
        env_config['reward_shape_mode'] = 'sparse'
        # For setting predicate bounds
        env_config['err_arg'] = 0
        # For setting maximum distance mode in predicate
        env_config['max_mode'] = True
        # For setting num_agents
        env_config['num_agents'] = num_agents
        # For setting time horizon
        env_config['horizon'] = horizon

        # Reward Scale Settings
        # To scale task monitor rewards
        env_config['limit_pred_val'] = True
        env_config['reward_scale'] = {'lb_scale': 2,  # tune.grid_search([1,5]),
                                      # tune.grid_search([2,3]),
                                      'lb_scale_factor': 2.0,
                                      'predicate_scaling': 5.0}

        # log(N) Scaling Settings
        # To scale training for MA systems
        # config[''][''] if 'logn_scaling' in config else 3.0
        #     log_n_factor = config['logn_scaling']['log_n_factor']
        env_config['limit_pred_val'] = True
        env_config['log_n_scaling'] = {'log_n_k': 2.0,
                                       'log_n_factor': 2.0}

        # Custom Model settings
        # For setting no monitor mode environment
        # Note: use custom model too
        no_monitor_mode = 'no_monitor' in env
        env_config['no_monitor'] = no_monitor_mode

        # Reward settings
        # To set TLTL Single Agent mode
        tltl_single_mode = 'tltl_sa' in env
        env_config['tltl_single_agent'] = tltl_single_mode
        # To set SPECTRL Single Agent mode
        spectrl_single_mode = 'spectrl_sa' in env
        env_config['spectrl_single_agent'] = spectrl_single_mode
        # To set distSPECTRL w/ Inline reward
        spectrl_inline_reward = 'inlineDS' in env
        env_config['inline_reward'] = spectrl_inline_reward
        # To set distSPECTRL w/ logNscaling Curriculum
        spectrl_logn = '_logN' in env
        env_config['spectrl_logn'] = spectrl_logn
        env_config['distributed_mode'] = spectrl_inline_reward or \
            not (spectrl_single_mode or tltl_single_mode)
        single_env = env_creator(env_config)
        # Save important functions/objects to env_config
        env_config.update(get_important_members(single_env))

    env_name = env + "_env"
    register_env(env_name, custom_env)

    # No need reward postprocessing since Inline
    MA_PolicyGraph = PPOTorchPolicy

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    def policy_mapping_fn(agent_id):
        return "learned"

    if centralized_ppo_mode:
        # Use Centralized Critic PPO
        from src.models.centralized_critic import CCPPOTorchPolicy, TorchCentralizedCriticModel
        ModelCatalog.register_custom_model(
            "cc_model", TorchCentralizedCriticModel)
        MA_PolicyGraph = CCPPOTorchPolicy

        from src.models.centralized_critic import CCTrainer
        agent_cls = CCTrainer
        algorithm = agent_cls
    elif no_monitor_mode:
        agent_cls = get_agent_class(algorithm)
    else:
        agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()
    # information for replay
    config['env_config']['func_create'] = tune.function(custom_env)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm
    config['env_config'].update(env_config)

    if 'rmachine' in env:
        # For better credit assignment in rmachine learning updates
        config['gamma'] = 0.999
        config['observation_filter'] = "NoFilter"
    else:
        # To allow the trajectory comparison properties of SPECTRL to hold
        config['gamma'] = 1.0
        config['observation_filter'] = "NoFilter"
        # '': "MeanStdFilter" if not 'nav' in env else ,  # Usable with InlineMDP?

    # Model config updates
    config["model"] = {"conv_filters": None,
                       "use_lstm": False,
                       "custom_model_config": {'num_agents': single_env.num_agents,
                                               'num_monitor_states': single_env.spec.monitor.n_states if hasattr(single_env, 'spec') and single_env.spec is not None
                                               else 0}
                       }
    config["framework"] = "torch"
    if centralized_ppo_mode:
        config["model"].update({"custom_model": "cc_model"})

    if eval_env_config is not None:
        config["evaluation_config"] = {
            'gamma': 1,
            'env_config': eval_env_config
        }

    # Calculate device configurations
    gpus_for_driver = 1 if use_gpu_for_driver else 0
    cpus_for_driver = 1 - gpus_for_driver
    eval_cpus = 1
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        # Limiting to 1 for parallel grid search
        spare_cpus = min(1, (num_cpus - cpus_for_driver - eval_cpus))
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
        "train_batch_size": train_batch_size,
        "horizon": horizon,
        # "lr": 3e-4,
        "lr_schedule":
            [[0, tune.grid_search([1e-5, 1e-3, 1e-4])],
             [20000000, hparams['lr_final']]],  # reduced from 2000000 since easier problem
        # "output": "/tmp/debug",
        # "output_compress_columns": [],
        "num_workers": num_workers,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
        "num_envs_per_worker": 2,
        "entropy_coeff": tune.grid_search([0, hparams['entropy_coeff']]),
        # "batch_mode": "complete_episodes",
        # Optional custom eval function.
        "evaluation_num_workers": eval_cpus,
        "evaluation_interval": 1,
        # Run 10 episodes each time evaluation runs.
        "evaluation_num_episodes": 5,
        "custom_eval_function": eval_fn,
        "callbacks": CustomCallbacks,
        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
                "learned": (MA_PolicyGraph, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },

    })
    return algorithm, env_name, config


def main(ununsed_argv):

    from ray.tune.logger import pretty_print

    from ray import tune
    from ray.tune import Experiment

    ray.init(num_cpus=FLAGS.num_cpus)
    hparams = default_params

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name

    alg_run, env_name, ppo_config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                          FLAGS.train_batch_size,
                                          FLAGS.num_cpus,
                                          FLAGS.num_gpus, FLAGS.num_agents,
                                          FLAGS.use_gpus_for_workers,
                                          FLAGS.use_gpu_for_driver,
                                          FLAGS.num_workers_per_device,
                                          FLAGS.spec_id, FLAGS.horizon)

    # Support PPO centralized version
    config = ppo_config
    config["env"] = env_name
    if not config['env_config']['distributed_mode']:
        config.pop("multiagent")

    e = Experiment(exp_name, run=alg_run,
                   stop={"evaluation/episode_reward_mean": 0.2,  # greater than zero means spec satisfied during evaluation
                         "training_iteration": FLAGS.training_iterations},
                   config=config,
                   checkpoint_freq=FLAGS.checkpoint_frequency
                   )
    callback_list = []
    if FLAGS.wandb:
        # NOTE: Fill in project and API key here and uncomment callback list
        api_str = ''
        project_str = ''
        # callback_list = [WandbLoggerCallback(project=api_str,
        #                                                 api_key=project_str)]

    tune.run_experiments([e],
                         callbacks=callback_list)

    ray.shutdown()


if __name__ == "__main__":
    main(0)
