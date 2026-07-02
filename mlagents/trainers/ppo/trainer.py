# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described in: https://arxiv.org/abs/1707.06347
from typing import cast, Type, Union, Dict, Any
from mlagents.trainers.torch_entities.networks_R import Cfg
import numpy as np
from mlagents.torch_utils import torch, default_device
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer_RLHF import OnPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.networks_R import SimpleActor_R
from mlagents.trainers.settings import OnPolicyHyperparamSettings
logger = get_logger(__name__)
TRAINER_NAME = "ppo"
class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            # batch_var = torch.var(x, axis=0)
            batch_var = torch.var(x, axis=0, unbiased=False)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
    @property
    def std(self):
        return torch.sqrt(self.var)
def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count
class PPOTrainer(OnPolicyTrainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training PPO model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.policy: TorchPolicy = None  # type: ignore
        self.num_timesteps = 0
        self.num_seedsteps = 0
        self.first_reward_train = 0
        self.thres_interaction = 5000
        self.num_feed = 1
        self.feed_type = 0
        self.re_update = 100
        self.traj_obsact = None
        self.traj_reward = None
        self.first_reward_train = 0
        self.num_interactions = 0
        self.metaworld_flag = False
        self.max_feed = 140000000
        self.total_feed = 0
        self.labeled_feedback = 0
        self.noisy_feedback = 0
        self.reward_batch = 50
        if self.policy:
            self.reward_batch = self.policy.actor.reward_model.mb_size
        self.unsuper_step = 0
        self.avg_train_true_return = 0
        self.size_segment = 25
        self.max_ep_len = 1000
        self.hyperparameters = cast(
            OnPolicyHyperparamSettings, self.trainer_settings.hyperparameters
        )
        self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=torch.device("cuda"))
    # def _process_trajectory(self, trajectory: Trajectory) -> None:
    #     """
    #     Takes a trajectory and processes it, putting it into the update buffer.
    #     Processing involves calculating value and advantage targets for model updating step.
    #     :param trajectory: The Trajectory tuple containing the steps to be processed.
    #     """
    #     super()._process_trajectory(trajectory)
    #     agent_id = trajectory.agent_id  # All the agents should have the same ID
    #     agent_buffer_trajectory = trajectory.to_agentbuffer()
    #     # Check if we used group rewards, warn if so.
    #     self._warn_if_group_reward(agent_buffer_trajectory)
    #     # Update the normalization
    #     if self.is_training:
    #         self.policy.actor.update_normalization(agent_buffer_trajectory)
    #         self.optimizer.critic.update_normalization(agent_buffer_trajectory)
    #     # Get all value estimates
    #     (
    #         value_estimates,
    #         value_next,
    #         value_memories,
    #     ) = self.optimizer.get_trajectory_value_estimates(
    #         agent_buffer_trajectory,
    #         trajectory.next_obs,
    #         trajectory.done_reached and not trajectory.interrupted,
    #     )
    #     if value_memories is not None:
    #         agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)
    #     for name, v in value_estimates.items():
    #         agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
    #             v
    #         )
    #         self._stats_reporter.add_stat(
    #             f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
    #             np.mean(v),
    #         )
    #     # Evaluate all reward functions
    #     self.collected_rewards["environment"][agent_id] += np.sum(
    #         agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
    #     )
    #     for name, reward_signal in self.optimizer.reward_signals.items():
    #         evaluate_result = (
    #             reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
    #         )
    #         agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
    #             evaluate_result
    #         )
    #         # Report the reward signals
    #         self.collected_rewards[name][agent_id] += np.sum(evaluate_result)
    #     # Compute GAE and returns
    #     tmp_advantages = []
    #     tmp_returns = []
    #     for name in self.optimizer.reward_signals:
    #         bootstrap_value = value_next[name]
    #         local_rewards = agent_buffer_trajectory[
    #             RewardSignalUtil.rewards_key(name)
    #         ].get_batch()
    #         local_value_estimates = agent_buffer_trajectory[
    #             RewardSignalUtil.value_estimates_key(name)
    #         ].get_batch()
    #         local_advantage = get_gae(
    #             rewards=local_rewards,
    #             value_estimates=local_value_estimates,
    #             value_next=bootstrap_value,
    #             gamma=self.optimizer.reward_signals[name].gamma,
    #             lambd=self.hyperparameters.lambd,
    #         )
    #         local_return = local_advantage + local_value_estimates
    #         # This is later use as target for the different value estimates
    #         agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
    #             local_return
    #         )
    #         agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
    #             local_advantage
    #         )
    #         tmp_advantages.append(local_advantage)
    #         tmp_returns.append(local_return)
    #     # Get global advantages
    #     global_advantages = list(
    #         np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
    #     )
    #     global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
    #     agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
    #     agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)
    #     self._append_to_update_buffer(agent_buffer_trajectory)
    #     # If this was a terminal trajectory, append stats and reset reward collection
    #     if trajectory.done_reached:
    #         self._update_end_episode_stats(agent_id, self.optimizer)
    def compute_state_entropy_(self, obs, full_obs, k):
        batch_size = 500
        with torch.no_grad():
            dists = []
            for idx in range(len(full_obs) // batch_size + 1):
                start = idx * batch_size
                end = (idx + 1) * batch_size
                dist = torch.norm(
                    obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
                )
                dists.append(dist)
            dists = torch.cat(dists, dim=1)
            knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
            state_entropy = knn_dists
        return state_entropy
    def _process_trajectory(self, trajectory: Trajectory) -> None:
            """
            Takes a trajectory and processes it, putting it into the update buffer.
            Processing involves calculating value and advantage targets for model updating step.
            :param trajectory: The Trajectory tuple containing the steps to be processed.
            """
            super()._process_trajectory(trajectory)
            agent_id = trajectory.agent_id  # All the agents should have the same ID
            agent_buffer_trajectory = trajectory.to_agentbuffer()
            self.num_seedsteps += agent_buffer_trajectory.num_experiences
            # Check if we used group rewards, warn if so.
            self._warn_if_group_reward(agent_buffer_trajectory)
            # Update the normalization
            if self.is_training:
                self.policy.actor.update_normalization(agent_buffer_trajectory)
                self.optimizer.critic.update_normalization(agent_buffer_trajectory)
            # Get all value estimates
            (
                value_estimates,
                value_next,
                value_memories,
            ) = self.optimizer.get_trajectory_value_estimates(
                agent_buffer_trajectory,
                trajectory.next_obs,
                trajectory.done_reached and not trajectory.interrupted,
            )
            if value_memories is not None:
                agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)
            for name, v in value_estimates.items():
                agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                    v
                )
                self._stats_reporter.add_stat(
                    f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                    np.mean(v),
                )
            # Evaluate all reward functions
            self.collected_rewards["environment"][agent_id] += np.sum(
                agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
            )
            for name, reward_signal in self.optimizer.reward_signals.items():
                evaluate_result = (
                    reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
                )
                agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                    evaluate_result
                )
                # Report the reward signals
                self.collected_rewards[name][agent_id] += np.sum(evaluate_result)
            # Compute GAE and returns
            tmp_advantages = []
            tmp_returns = []
            if len(self.optimizer.reward_signals) > 1.5:
                assert len(self.optimizer.reward_signals)
            for name in self.optimizer.reward_signals:
                bootstrap_value = value_next[name]
                local_rewards = agent_buffer_trajectory[
                    RewardSignalUtil.rewards_key(name)
                ].get_batch()
                local_value_estimates = agent_buffer_trajectory[
                    RewardSignalUtil.value_estimates_key(name)
                ].get_batch()
                if self.num_seedsteps > self.hyperparameters.buffer_size + Cfg.NUM_SEED_STEPS.value + Cfg.NUM_UNSUP_STEPS.value:
                    n_obs = len(self.policy.behavior_spec.observation_specs)
                    current_obs = ObsUtil.from_buffer(agent_buffer_trajectory, n_obs)
                    # obs = [ModelUtils.list_to_tensor(_) for _ in current_obs]
                    # obs = current_obs
                    act = agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION]
                    obs = []
                    for j in range(0, len(current_obs[0])):
                        obs.append([])
                        for i in range(0, len(current_obs)):
                            obs[j].extend(current_obs[i][j])
                    # print("obs:", len(obs),len(obs[0]))
                    # print("act:", len(act),len(act[0]))
                    obsact = np.concatenate((obs, act), axis=-1) # num_env x (obs+act)
                    # print("obsact1:", len(obsact),len(obsact[0]))
                    obsact = np.expand_dims(obsact, axis=1) # num_env x 1 x (obs+act) 
                    # print("obsact2:", len(obsact),len(obsact[0]))
                    pred_reward = self.policy.actor.reward_model.r_hat_batch(obsact)
                    pred_reward = pred_reward.reshape(-1)
                    # print("pred_reward", len(pred_reward))
                    # print("local_rewards", len(local_rewards))
                    if self.traj_obsact is None:
                        self.traj_obsact = obsact
                        self.traj_reward = local_rewards
                    else:
                        self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                        self.traj_reward = np.concatenate((self.traj_reward, local_rewards), axis=1)
                    self.num_timesteps += len(pred_reward)
                    self.num_interactions += len(pred_reward)
                    # local_rewards = pred_reward
                    # local_advantage = get_gae(
                    #     rewards=local_rewards,
                    #     value_estimates=local_value_estimates,
                    #     value_next=bootstrap_value,
                    #     gamma=self.optimizer.reward_signals[name].gamma,
                    #     lambd=self.hyperparameters.lambd,
                    # )
                    local_advantage = get_gae(
                        rewards=pred_reward,
                        value_estimates=local_value_estimates,
                        value_next=bootstrap_value,
                        gamma=self.optimizer.reward_signals[name].gamma,
                        lambd=self.hyperparameters.lambd,
                    )
                    self.policy.actor.reward_model.add_data_batch_R(obs, act, self.traj_reward, trajectory.done_reached)
                                # reset traj
                    self.traj_obsact, self.traj_reward = None, None
                    # train reward using random data
                    if self.first_reward_train == 0:
                        self.learn_reward()
                        self.first_reward_train = 1
                        self.num_interactions = 0
                    else:
                        if self.num_interactions >= self.thres_interaction and self.total_feed < self.max_feed:
                            self.learn_reward()
                            self.num_interactions = 0
                elif self.num_seedsteps > self.hyperparameters.buffer_size + Cfg.NUM_SEED_STEPS.value: 
                    self._append_to_update_buffer_all(agent_buffer_trajectory)
                    batch_all = self.update_buffer_all
                    # batch_size=1024
                    K=5
                    # tail = self.hyperparameters.buffer_size - 1
                    # idxs = np.random.randint(0,s
                    #                         tail,
                    #                         size=batch_size)
                    # batch = batch_all.idx_mini_batch(idxs, -1 * tail)
                    # rewards = {}
                    # for name, signal in self.reward_signals.items():
                    #     rewards[name] = ModelUtils.list_to_tensor((
                    #         signal.evaluate(batch_all) * signal.strength
                    #     ))
                    # returns = {}
                    # for name in self.reward_signals:
                    #     returns[name] = ModelUtils.list_to_tensor(
                    #         batch[RewardSignalUtil.returns_key(name)]
                    #     )
                    n_obs = len(self.policy.behavior_spec.observation_specs)
                    current_obs_all = ObsUtil.from_buffer(batch_all, n_obs)
                    # Convert to tensors
                    current_obs_all = [ModelUtils.list_to_tensor(obs) for obs in current_obs_all] 
                    current_obs = ObsUtil.from_buffer(agent_buffer_trajectory, n_obs)
                    # Convert to tensors
                    current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
                    # current_obs_cut = [t[(-1 * tail):] for t in current_obs_all]
                    # if batch_size >= batch.num_experiences:
                    current_obs = torch.cat(current_obs, dim=1)
                    full_obs = torch.cat(current_obs_all, dim=1)
                    # else:
                    #     full_obs = torch.cat(current_obs_all, dim=1)
                    # print("current_obs[(-1 * batch_size):]", len(current_obs[:][(-1 * batch_size):][0]))
                    full_obs = torch.as_tensor(full_obs, device = torch.device("cuda"))
                    current_obs = torch.as_tensor(current_obs, device = torch.device("cuda"))
                    # # idxs = np.array(idxs)
                    # # print("full_obs", len(full_obs))
                    # random_obs = full_obs[idxs]
                    # print("shape", current_obs.shape, full_obs.shape)
                    state_entropy = self.compute_state_entropy_(current_obs, full_obs, 5)
                    self.s_ent_stats.update(state_entropy)
                    norm_state_entropy = state_entropy / self.s_ent_stats.std
                    normalize_state_entropy = True
                    if normalize_state_entropy or True:
                        state_entropy = norm_state_entropy
                    pred_reward = norm_state_entropy.reshape(-1).data.cpu().numpy()
                    # rewards = {}
                    # 
                    # for name in self.reward_signals:
                    #     # print("rewards[name]", len(rewards[name]))
                    #     # print("state_entropy", len(state_entropy))
                    #     returns[name] = state_entropy
                    # local_rewards = state_entropy
                    local_advantage = get_gae(
                            rewards=pred_reward,
                            value_estimates=local_value_estimates,
                            value_next=bootstrap_value,
                            gamma=self.optimizer.reward_signals[name].gamma,
                            lambd=self.hyperparameters.lambd,
                        )
                else:
                    local_advantage = get_gae(
                            rewards=local_rewards,
                            value_estimates=local_value_estimates,
                            value_next=bootstrap_value,
                            gamma=self.optimizer.reward_signals[name].gamma,
                            lambd=self.hyperparameters.lambd,
                        )
                local_return = local_advantage + local_value_estimates
                # This is later use as target for the different value estimates
                agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                    local_return
                )
                agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                    local_advantage
                )
                tmp_advantages.append(local_advantage)
                tmp_returns.append(local_return)
            # Get global advantages
            global_advantages = list(
                np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
            )
            global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
            agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
            agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)
            self._append_to_update_buffer(agent_buffer_trajectory)
            # If this was a terminal trajectory, append stats and reset reward collection
            if trajectory.done_reached:
                self._update_end_episode_stats(agent_id, self.optimizer)
    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPOOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore
    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and PPO hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls: Union[Type[SimpleActor_R], Type[SharedActorCritic]] = SimpleActor_R
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [
                key.value for key, _ in reward_signal_configs.items()
            ]
            actor_cls = SharedActorCritic
            actor_kwargs.update({"stream_names": reward_signal_names})
        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        return policy
    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """
        return self.policy
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    def learn_reward(
        self) -> None:
        # update margin
        new_margin = np.mean(self.avg_train_true_return) * (self.size_segment / self.max_ep_len)
        self.policy.actor.reward_model.set_teacher_thres_skip(new_margin)
        self.policy.actor.reward_model.set_teacher_thres_equal(new_margin)
        if self.first_reward_train == 0:
            labeled_queries = self.policy.actor.reward_model.uniform_sampling()
        else:
            if self.feed_type == 0:
                labeled_queries = self.policy.actor.reward_model.uniform_sampling()
            elif self.feed_type == 1:
                labeled_queries = self.policy.actor.reward_model.disagreement_sampling()
            elif self.feed_type == 2:
                labeled_queries = self.policy.actor.reward_model.entropy_sampling()
            else:
                raise NotImplementedError
        if labeled_queries == 0:
            print("traj is not enough, skip training")
            return
        self.total_feed += self.policy.actor.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        # update reward
        for epoch in range(self.re_update):
            if self.policy.actor.reward_model.teacher_eps_equal > 0:
                train_acc = self.policy.actor.reward_model.train_soft_reward()
            else:
                train_acc = self.policy.actor.reward_model.train_reward()
            total_acc = np.mean(train_acc)
            if total_acc > 0.97:
                break
        print("Reward function is updated!! ACC: " + str(total_acc))
        logger = get_logger(__name__)
