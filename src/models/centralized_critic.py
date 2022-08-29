"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
import numpy as np

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
try:
    from ray.rllib.agents.ppo.ppo_torch_policy import (
        KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss)
    ray_version_greater_than15 = False
except ImportError:
    # Newer ray version
    ray_version_greater_than15 = True
    from ray.rllib.agents.maml.maml_torch_policy import KLCoeffMixin as TorchKLCoeffMixin
    torch_loss = PPOTorchPolicy.loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import  try_import_torch
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        num_agents = model_config['custom_model_config']['num_agents']
        obs_dim = obs_space.shape[0]
        other_agent_obs_dim = obs_dim * (num_agents - 1)
        action_dim = action_space.shape[0] * (num_agents - 1)
        self.other_agent_obs_dim = other_agent_obs_dim
        self.other_agent_action_dim = action_dim

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs,
                             model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = obs_dim + other_agent_obs_dim + \
            action_dim  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        input_ = torch.cat([
            obs, opponent_obs, opponent_actions
            # torch.nn.functional.one_hot(.long(), 2).float()
        ], 1)
        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batches = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = np.hstack([opponent_batch[1][SampleBatch.CUR_OBS]
                                                for opponent_batch in opponent_batches])
        sample_batch[OPPONENT_ACTION] = np.hstack([opponent_batch[1][SampleBatch.ACTIONS]
                                                   for opponent_batch in opponent_batches])

        if pytorch:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[SampleBatch.CUR_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_OBS], policy.device),
                convert_to_torch_tensor(
                    sample_batch[OPPONENT_ACTION], policy.device)) \
                .cpu().detach().numpy()
        else:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                sample_batch[SampleBatch.CUR_OBS], sample_batch[OPPONENT_OBS],
                sample_batch[OPPONENT_ACTION])
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros((sample_batch[SampleBatch.ACTIONS].shape[0],
                                               policy.model.other_agent_obs_dim))
        sample_batch[OPPONENT_ACTION] = np.zeros((sample_batch[SampleBatch.ACTIONS].shape[0],
                                                  policy.model.other_agent_action_dim))
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    env_config = policy.config['env_config']
    shaped_rw_batch = sample_batch

    train_batch = compute_advantages(
        shaped_rw_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = torch_loss
    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }


if not ray_version_greater_than15:
    CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="CCPPOTorchPolicy",
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic,
        before_init=setup_torch_mixins,
        mixins=[
            TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin
        ])

    def get_policy_class(config):
        return CCPPOTorchPolicy
    CCTrainer = PPOTrainer.with_updates(
        name="CCPPOTrainer",
        default_policy=CCPPOTorchPolicy,
        get_policy_class=get_policy_class,
    )

else:
    # new ray version
    class CCPPOTorchPolicy(PPOTorchPolicy):
        def __init__(self, observation_space, action_space, config):
            super().__init__(observation_space, action_space, config)
            self.compute_central_vf = self.model.central_value_function

        @override(PPOTorchPolicy)
        def loss(self, model, dist_class, train_batch):
            return loss_with_central_critic(self, model, dist_class, train_batch)

        @override(PPOTorchPolicy)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

    class CCTrainer(PPOTrainer):
        @override(PPOTrainer)
        def get_default_policy_class(self, config):
            return CCPPOTorchPolicy
