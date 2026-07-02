# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines
from collections import defaultdict
from typing import Dict, cast
import os
from mlagents.trainers.torch_entities.networks_R import SimpleActor_R
import numpy as np
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import timed
from mlagents.trainers.buffer import RewardSignalUtil
from mlagents.trainers.policy import Policy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.torch_entities.networks_R import Cfg, RewardModel
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.agent_action import AgentAction
logger = get_logger(__name__)
from collections import deque
BUFFER_TRUNCATE_PERCENT = 0.8
class OffPolicyTrainer_R(RLTrainer):
    """
    The Trainer is an implementation of the SAC algorithm, with support
    for discrete actions and recurrent networks.
    """
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
        Responsible for collecting experiences and training an off-policy model.
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
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap,
        )
        self.seed = seed
        self.policy: Policy = None  # type: ignore
        self.optimizer: TorchOptimizer = None  # type: ignore
        self.hyperparameters: OffPolicyHyperparamSettings = cast(
            OffPolicyHyperparamSettings, trainer_settings.hyperparameters
        )
        # Cfg = Cfg()
        self.total_feedback = 0
        self.labeled_feedback = 0
        self._step = 0
        self.episode = 0 
        self.episode_reward = 0
        self.done = True
        self.true_episode_reward = 0
        # Don't divide by zero
        self.update_steps = 1
        self.reward_signal_update_steps = 1
        self.interact_count = 0
        self.steps_per_update = self.hyperparameters.steps_per_update
        self.reward_signal_steps_per_update = (
            self.hyperparameters.reward_signal_steps_per_update
        )
        self.work_dir = self.artifact_path
        self.checkpoint_replay_buffer = self.hyperparameters.save_replay_buffer
        self.avg_train_true_return = deque([], maxlen=10) 
        self.buffer_len = 0
        self.train_acc = 0
        self.avg_train_true_return
    def _checkpoint(self) -> ModelCheckpoint:
        """
        Writes a checkpoint model to memory
        Overrides the default to save the replay buffer.
        """
        ckpt = super()._checkpoint()
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
        return ckpt
    def save_model(self) -> None:
        """
        Saves the final training model to memory
        Overrides the default to save the replay buffer.
        """
        super().save_model()
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
    def save_replay_buffer(self) -> None:
        """
        Save the training buffer's update buffer to a pickle file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Saving Experience Replay Buffer to {filename}...")
        with open(filename, "wb") as file_object:
            self.update_buffer.save_to_file(file_object)
            logger.info(
                f"Saved Experience Replay Buffer ({os.path.getsize(filename)} bytes)."
            )
    def load_replay_buffer(self) -> None:
        """
        Loads the last saved replay buffer from a file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Loading Experience Replay Buffer from {filename}...")
        with open(filename, "rb+") as file_object:
            self.update_buffer.load_from_file(file_object)
        logger.debug(
            "Experience replay buffer has {} experiences.".format(
                self.update_buffer.num_experiences
            )
        )
    def _is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not _update_policy() can be run
        """
        return (
            self.update_buffer.num_experiences >= self.hyperparameters.batch_size
            and self._step >= self.hyperparameters.buffer_init_steps
        )
    def maybe_load_replay_buffer(self):
        # Load the replay buffer if load
        if self.load and self.checkpoint_replay_buffer:
            try:
                self.load_replay_buffer()
            except (AttributeError, FileNotFoundError):
                logger.warning(
                    "Replay buffer was unable to load, starting from scratch."
                )
            logger.debug(
                "Loaded update buffer with {} sequences".format(
                    self.update_buffer.num_experiences
                )
            )
    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy
        self.optimizer = self.create_optimizer()
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)
        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()
        # Needed to resume loads properly
        self._step = policy.get_current_step()
        # Assume steps were updated at the correct ratio before
        self.update_steps = int(max(1, self._step / self.steps_per_update))
        self.reward_signal_update_steps = int(
            max(1, self._step / self.reward_signal_steps_per_update)
        )
    @timed
    def _update_policy(self) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        has_updated = False
        self.cumulative_returns_since_policy_update.clear()
        n_sequences = max(
            int(self.hyperparameters.batch_size / self.policy.sequence_length), 1
        )
        batch_update_stats: Dict[str, list] = defaultdict(list)
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.update_steps > self.steps_per_update:
            logger.debug(f"Updating SAC policy at step {self._step}")
            buffer = self.update_buffer
            if self.update_buffer.num_experiences >= self.hyperparameters.batch_size:
                sampled_minibatch = buffer.sample_mini_batch(
                    self.hyperparameters.batch_size,
                    sequence_length=self.policy.sequence_length,
                )
                # Get rewards for each reward
                for name, signal in self.optimizer.reward_signals.items():
                    sampled_minibatch[RewardSignalUtil.rewards_key(name)] = (
                        signal.evaluate(sampled_minibatch) * signal.strength
                    )
                update_stats = self.optimizer.update(sampled_minibatch, n_sequences)
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)
                self.update_steps += 1
                for stat, stat_list in batch_update_stats.items():
                    self._stats_reporter.add_stat(stat, np.mean(stat_list))
                has_updated = True
            if self.optimizer.bc_module:
                update_stats = self.optimizer.bc_module.update()
                for stat, val in update_stats.items():
                    self._stats_reporter.add_stat(stat, val)
        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        if self.update_buffer.num_experiences > self.hyperparameters.buffer_size:
            self.update_buffer.truncate(
                int(self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
            )
        # TODO: revisit this update
        self._update_reward_signals()
        return has_updated
    @timed
    def _update_policy_R(self) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        self.policy : TorchPolicy
        self.policy.actor : SimpleActor_R
        self.policy.actor.reward_model : RewardModel
        has_updated = False
        self.cumulative_returns_since_policy_update.clear()
        n_sequences = max(
            int(self.hyperparameters.batch_size / self.policy.sequence_length), 1
        )
        use_pebble = 1
        saving_bool = False
        batch_update_stats: Dict[str, list] = defaultdict(list)
        buffer = self.update_buffer
        buffer_mini = self.update_buffer.make_mini_batch(self.buffer_len, self.update_buffer.num_experiences)
        self.buffer_len = self.update_buffer.num_experiences
        self.episode_reward = 0
        self.true_episode_reward = 0
        buffer_reward_update = True
        # Get rewards for each reward
        for name, signal in self.optimizer.reward_signals.items():
            buffer_mini[RewardSignalUtil.rewards_key(name)] = (
                signal.evaluate(buffer_mini) * signal.strength
            )
            rewards = buffer_mini[RewardSignalUtil.rewards_key(name)]
        next_obs : list    
        reward : float
        done : float
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(buffer_mini, n_obs)
        obs_all = []
        for j in range(0, len(current_obs[0])):
            obs_all.append([])
            for i in range(0, len(current_obs)):
                obs_all[j].extend(current_obs[i][j])
        next_obs = ObsUtil.from_buffer_next(buffer_mini, n_obs)
        next_obs_all = []
        for j in range(0, len(next_obs[0])):
            next_obs_all.append([])
            for i in range(0, len(next_obs)):
                next_obs_all[j].extend(next_obs[i][j])
        actions = list(buffer_mini[BufferKey.CONTINUOUS_ACTION])
        # self.buffer_len = len(actions) - 1
            # [0.  0.1 0.  ... 0.  0.  0. ]
        dones = list(buffer_mini[BufferKey.DONE])
        # self.policy.actor.reward_model.clear_data()
        # print("dones", dones)
        if use_pebble == 1:
            for j in range(0, len(dones)):
                obs = obs_all[j]
                next_obs = next_obs_all[j]
                action = actions[j]
                reward = rewards[j]
                done = float(dones[j])
                # print("info:", obs, next_obs, action, reward, done)
                # next_obs, reward, done, extra = self.buffer_collect(sampled_minibatch)
                # reward_hat = self.policy.actor.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                # allow infinite bootstrap
                # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
                done_no_max = done
                # self.episode_reward += reward_hat
                self.true_episode_reward += reward
                # if self.log_success:
                #     episode_success = max(episode_success, extra['success'])
                # adding data to the reward training data
                self.policy.actor.reward_model.add_data(obs, action, reward, done)
                # self.replay_buffer.add(
                #     obs, action, reward_hat, 
                #     next_obs, done, done_no_max)
                # break
                # obs = next_obs
                # episode_step += 1
            saving_bool = False
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.update_steps > self.steps_per_update:
            logger.debug(f"Updating SAC policy at step {self._step}")
            if self.update_buffer.num_experiences >= self.hyperparameters.batch_size:
                sampled_minibatch = buffer.sample_mini_batch(
                    self.hyperparameters.batch_size,
                    sequence_length=self.policy.sequence_length,
                )
                for name, signal in self.optimizer.reward_signals.items():
                    sampled_minibatch[RewardSignalUtil.rewards_key(name)] = (
                        signal.evaluate(sampled_minibatch) * signal.strength
                    )
                # print(sampled_minibatch.num_experiences)
                # n_obs = len(self.policy.behavior_spec.observation_specs)
                obses_ori = ObsUtil.from_buffer(sampled_minibatch, n_obs)
                actions_ori = list(sampled_minibatch[BufferKey.CONTINUOUS_ACTION])
                obs_sample = []
                for j in range(0, len(obses_ori[0])):
                    obs_sample.append([])
                    for i in range(0, len(obses_ori)):
                        obs_sample[j].extend(obses_ori[i][j])
                # if self.update_steps == (Cfg.NUM_SEED_STEPS + Cfg.NUM_UNSUP_STEPS):
                if self.update_steps < Cfg.NUM_SEED_STEPS.value:
                    # no need to update policy
                    pass
                if self.update_steps == Cfg.NUM_SEED_STEPS.value + Cfg.NUM_UNSUP_STEPS.value:
                    # update schedule
                    saving_bool = True
                    if Cfg.REWARD_SCHEDULE.value == 1:
                        frac = (Cfg.NUM_TRAIN_STEPS.value-self.update_steps) / Cfg.NUM_TRAIN_STEPS.value
                        if frac == 0:
                            frac = 0.01
                    elif Cfg.REWARD_SCHEDULE.value == 2:
                        frac = Cfg.NUM_TRAIN_STEPS.value / (Cfg.NUM_TRAIN_STEPS.value-self.update_steps +1)
                    else:
                        frac = 1
                    self.policy.actor.reward_model.change_batch(frac)
                    # update margin --> not necessary / will be updated soon
                    # print("update margin")
                    new_margin = np.mean(self.avg_train_true_return) * (Cfg.SEGMENT.value / Cfg._MAX_EPISODE_STEPS.value)
                    self.policy.actor.reward_model.set_teacher_thres_skip(new_margin)
                    self.policy.actor.reward_model.set_teacher_thres_equal(new_margin)
                    # first learn reward
                    if use_pebble == 1:
                        self.train_acc = self.learn_reward(first_flag=1)
                        batch_update_stats["Losses/ACC Loss"].append(self.train_acc)
                    # relabel buffer
                    # sampled_minibatch.relabel_with_predictor(self.policy.actor.reward_model, self.optimizer.reward_signals, obs_sample, actions_ori)
                    # reset Q due to unsuperivsed exploration
                    # self.agent.reset_critic()
                    # update agent
                    # self.agent.update_after_reset(
                    #     self.replay_buffer, self.logger, self.step, 
                    #     gradient_update=Cfg.reset_update, 
                    #     policy_update=True)
                    # reset interact_count
                    self.interact_count = 0
                # elif self.update_steps > Cfg.NUM_SEED_STEPS + Cfg.NUM_UNSUP_STEPS:
                elif self.update_steps > Cfg.NUM_SEED_STEPS.value + Cfg.NUM_UNSUP_STEPS.value:
                    # update reward function
                    if self.total_feedback < Cfg.MAX_FEEDBACK.value and use_pebble == 1:
                        if self.interact_count == Cfg.NUM_INTERACT.value:
                            # update schedule
                            if Cfg.REWARD_SCHEDULE.value == 1:
                                frac = (Cfg.NUM_TRAIN_STEPS.value-self.update_steps) / Cfg.NUM_TRAIN_STEPS.value
                                if frac == 0:
                                    frac = 0.01
                            elif Cfg.REWARD_SCHEDULE.value == 2:
                                frac = Cfg.NUM_TRAIN_STEPS.value / (Cfg.NUM_TRAIN_STEPS.value-self.update_steps +1)
                            else:
                                frac = 1
                            self.policy.actor.reward_model.change_batch(frac)
                            # update margin --> not necessary / will be updated soon
                            new_margin = np.mean(self.avg_train_true_return) * (Cfg.SEGMENT.value / Cfg._MAX_EPISODE_STEPS.value)
                            self.policy.actor.reward_model.set_teacher_thres_skip(new_margin * Cfg.TEACHER_EPS_SKIP.value)
                            self.policy.actor.reward_model.set_teacher_thres_equal(new_margin * Cfg.TEACHER_EPS_EQUAL.value)
                            # corner case: new total feed > max feed
                            if self.policy.actor.reward_model.mb_size + self.total_feedback > Cfg.MAX_FEEDBACK.value:
                                self.policy.actor.reward_model.set_batch(Cfg.MAX_FEEDBACK.value - self.total_feedback)
                            saving_bool = True
                            self.train_acc = self.learn_reward()
                            batch_update_stats["Losses/ACC Loss"].append(self.train_acc)
                            self.interact_count = 0
                    if use_pebble == 1:
                        sampled_minibatch.relabel_with_predictor(self.policy.actor.reward_model, self.optimizer.reward_signals, obs_sample, actions_ori)
                    # print("reward:", sampled_minibatch[RewardSignalUtil.rewards_key(name)])
                    # print("self.optimizer1", self.optimizer)
                    update_stats, current_obs, next_obs, actions_, rewards_, dones_ = self.optimizer.update(sampled_minibatch, n_sequences)
                    for stat_name, value in update_stats.items():
                        batch_update_stats[stat_name].append(value)
                    for stat, stat_list in batch_update_stats.items():
                        self._stats_reporter.add_stat(stat, np.mean(stat_list))
                        # print("stat", stat, stat_list)
                    has_updated = True
                    self.interact_count += 1    
                elif self.update_steps > Cfg.NUM_SEED_STEPS.value:
                    # self.policy.actor.update_state_ent(self.replay_buffer, self.logger, self.step, 
                    #                         gradient_update=1, K=5)
                    if buffer_reward_update:
                        buffer_reward_update = False
                    update_stats = self.optimizer.update_state_ent(batch_all=buffer, batch_size=1024, gradient_update=1, K=5, normalize_state_entropy=True)
                    # update_stats = self.optimizer.update(sampled_minibatch, n_sequences)
                    # update_stats, current_obs, next_obs, actions_, rewards_, dones_ = self.optimizer.update(sampled_minibatch, n_sequences)
                    # print("self.optimizer2", self.optimizer)
                    for stat_name, value in update_stats.items():
                        batch_update_stats[stat_name].append(value)
                    for stat, stat_list in batch_update_stats.items():
                        self._stats_reporter.add_stat(stat, np.mean(stat_list))
                    has_updated = True
                    # pass
                # print("self.update_steps:", self.update_steps)
                self.update_steps += 1
                if self.optimizer.bc_module:
                    update_stats = self.optimizer.bc_module.update()
                    for stat, val in update_stats.items():
                        self._stats_reporter.add_stat(stat, val)            
                #     update_stats, current_obs, next_obs, actions_, rewards_, dones_ = self.optimizer.update(sampled_minibatch, n_sequences)
                #     for stat_name, value in update_stats.items():
                #         batch_update_stats[stat_name].append(value)
                #     # self.update_steps += 1
                #     for stat, stat_list in batch_update_stats.items():
                #         self._stats_reporter.add_stat(stat, np.mean(stat_list))
                #     has_updated = True
                #     if self.optimizer.bc_module:
                #         update_stats = self.optimizer.bc_module.update()
                #         for stat, val in update_stats.items():
                #             self._stats_reporter.add_stat(stat, val)
                # # unsupervised exploration
                # elif self.step > Cfg.num_seed_steps:
                #     self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                #                                 gradient_update=1, K=Cfg.topK)
                # self.step += 1
                # print("sampled_minibatch ori:", sampled_minibatch[ObservationKeyPrefix.OBSERVATION, 1][0])
                # for sampled_minibatch_f1 in sampled_minibatch:
                #     print("sampled_minibatch ori:", sampled_minibatch_f1)
                    # print("content:", sampled_minibatch_f2)
            # reward model
        if saving_bool: 
            self.policy.actor.reward_model.save(self.work_dir, self.update_steps)
            saving_bool = False
            # print("reward model out", self.work_dir)
        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        if self.update_buffer.num_experiences > self.hyperparameters.buffer_size:
            self.update_buffer.truncate(
                int(self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
            )
            # print("truncate",self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
        # TODO: revisit this update
        self._update_reward_signals()
        return has_updated
    def _update_reward_signals(self) -> None:
        """
        Iterate through the reward signals and update them. Unlike in PPO,
        do it separate from the policy so that it can be done at a different
        interval.
        This function should only be used to simulate
        http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times. Normally, the reward signal
        and policy are updated in parallel.
        """
        buffer = self.update_buffer
        batch_update_stats: Dict[str, list] = defaultdict(list)
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.reward_signal_update_steps > self.reward_signal_steps_per_update:
            # Get minibatches for reward signal update if needed
            minibatch = buffer.sample_mini_batch(
                self.hyperparameters.batch_size,
                sequence_length=self.policy.sequence_length,
            )
            update_stats = self.optimizer.update_reward_signals(minibatch)
            for stat_name, value in update_stats.items():
                batch_update_stats[stat_name].append(value)
            self.reward_signal_update_steps += 1
            for stat, stat_list in batch_update_stats.items():
                self._stats_reporter.add_stat(stat, np.mean(stat_list))
    def learn_reward(self, first_flag=0):
            # debug time
            # print("learn_reward()")
            # get feedbacks
            labeled_queries, noisy_queries = 0, 0
            if first_flag == 1:
                # if it is first time to get feedback, need to use random sampling
                labeled_queries = self.policy.actor.reward_model.uniform_sampling()
            else:
                if Cfg.FEED_TYPE.value == 0:
                    labeled_queries = self.policy.actor.reward_model.uniform_sampling()
                elif Cfg.FEED_TYPE.value == 1:
                    labeled_queries = self.policy.actor.reward_model.disagreement_sampling()
                elif Cfg.FEED_TYPE.value == 2:
                    labeled_queries = self.policy.actor.reward_model.entropy_sampling()
                elif Cfg.FEED_TYPE.value == 3:
                    labeled_queries = self.policy.actor.reward_model.kcenter_sampling()
                elif Cfg.FEED_TYPE.value == 4:
                    labeled_queries = self.policy.actor.reward_model.kcenter_disagree_sampling()
                elif Cfg.FEED_TYPE.value == 5:
                    labeled_queries = self.policy.actor.reward_model.kcenter_entropy_sampling()
                else:
                    raise NotImplementedError
            self.total_feedback += self.policy.actor.reward_model.mb_size
            self.labeled_feedback += labeled_queries
            train_acc = 0
            if self.labeled_feedback > 0:
                # update reward
                for epoch in range(Cfg.REWARD_UPDATE.value):
                    if Cfg.LABEL_MARGIN.value > 0 or Cfg.TEACHER_EPS_EQUAL.value > 0:
                        train_acc = self.policy.actor.reward_model.train_soft_reward()
                    else:
                        train_acc = self.policy.actor.reward_model.train_reward()
                    total_acc = np.mean(train_acc)
                    if total_acc > 0.97:
                        break
            print("Reward function is updated!! ACC: " + str(total_acc))
            return total_acc
    def buffer_collect(sampled_minibatch : AgentBuffer):
        for sampled_mini in sampled_minibatch:
            pass
