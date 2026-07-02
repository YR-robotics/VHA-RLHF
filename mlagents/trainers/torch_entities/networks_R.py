from typing import Callable, List, Dict, Tuple, Optional, Union, Any
import abc
from mlagents.torch_utils import torch, nn
from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.settings import NetworkSettings, EncoderType, ConditioningType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.layers import LSTM, LinearEncoder
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.conditioning import ConditionalEncoder
from mlagents.trainers.torch_entities.attention import (
    EntityEmbedding,
    ResidualSelfAttention,
    get_zero_entities_mask,
)
from mlagents.trainers.exception import UnityTrainerException
import numpy as np
import enum
import time
import torch.nn.functional as F
device = 'cuda'
ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]
EPSILON = 1e-7
class Cfg(enum.Enum):
    AGENT= "sac"
    EXPERIMENT= "PEBBLE"
    SEGMENT= 20
    ACTIVATION= "tanh"
    NUM_SEED_STEPS= 0
    NUM_UNSUP_STEPS= 500000
    NUM_INTERACT= 50
    REWARD_LR= 0.0003
    REWARD_BATCH= 128
    REWARD_UPDATE= 200
    FEED_TYPE= 0
    RESET_UPDATE= 100
    TOPK= 5
    ENSEMBLE_SIZE= 3
    MAX_FEEDBACK= 14000000
    LARGE_BATCH= 10
    LABEL_MARGIN= 0.0
    TEACHER_BETA= -1
    TEACHER_GAMMA= 1
    TEACHER_EPS_MISTAKE= 0
    TEACHER_EPS_SKIP= 0
    TEACHER_EPS_EQUAL= 0
    REWARD_SCHEDULE= 0
    NUM_TRAIN_STEPS= 1E6
    REPLAY_BUFFER_CAPACITY= NUM_TRAIN_STEPS
    EVAL_FREQUENCY= 10000
    NUM_EVAL_EPISODES= 10
    DEVICE= "cuda"
    LOG_FREQUENCY= 10000
    LOG_SAVE_TB= True
    _MAX_EPISODE_STEPS = 1000
    SAVE_VIDEO= False
    SEED= 1
    ENV= "dog_stand"
    GRADIENT_UPDATE= 1
class ObservationEncoder(nn.Module):
    ATTENTION_EMBEDDING_SIZE = 128                                            
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        normalize: bool = False,
    ):
        """
        Returns an ObservationEncoder that can process and encode a set of observations.
        Will use an RSA if needed for variable length observations.
        """
        super().__init__()
        self.processors, self.embedding_sizes = ModelUtils.create_input_processors(
            observation_specs,
            h_size,
            vis_encode_type,
            self.ATTENTION_EMBEDDING_SIZE,
            normalize=normalize,
        )
        self.rsa, self.x_self_encoder = ModelUtils.create_residual_self_attention(
            self.processors, self.embedding_sizes, self.ATTENTION_EMBEDDING_SIZE
        )
        if self.rsa is not None:
            total_enc_size = sum(self.embedding_sizes) + self.ATTENTION_EMBEDDING_SIZE
        else:
            total_enc_size = sum(self.embedding_sizes)
        self.normalize = normalize
        self._total_enc_size = total_enc_size
        self._total_goal_enc_size = 0
        self._goal_processor_indices: List[int] = []
        for i in range(len(observation_specs)):
            if observation_specs[i].observation_type == ObservationType.GOAL_SIGNAL:
                self._total_goal_enc_size += self.embedding_sizes[i]
                self._goal_processor_indices.append(i)
    @property
    def total_enc_size(self) -> int:
        """
        Returns the total encoding size for this ObservationEncoder.
        """
        return self._total_enc_size
    @property
    def total_goal_enc_size(self) -> int:
        """
        Returns the total goal encoding size for this ObservationEncoder.
        """
        return self._total_goal_enc_size
    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, len(self.processors))
        for vec_input, enc in zip(obs, self.processors):
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(vec_input.to_ndarray()))
    def copy_normalization(self, other_encoder: "ObservationEncoder") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_encoder.processors):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations using a list of processors and an RSA.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []
        var_len_processor_inputs: List[Tuple[nn.Module, torch.Tensor]] = []
        for idx, processor in enumerate(self.processors):
            if not isinstance(processor, EntityEmbedding):
                obs_input = inputs[idx]
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            else:
                var_len_processor_inputs.append((processor, inputs[idx]))
        if len(encodes) != 0:
            encoded_self = torch.cat(encodes, dim=1)
            input_exist = True
        else:
            input_exist = False
        if len(var_len_processor_inputs) > 0 and self.rsa is not None:
            masks = get_zero_entities_mask([p_i[1] for p_i in var_len_processor_inputs])
            embeddings: List[torch.Tensor] = []
            processed_self = (
                self.x_self_encoder(encoded_self)
                if input_exist and self.x_self_encoder is not None
                else None
            )
            for processor, var_len_input in var_len_processor_inputs:
                embeddings.append(processor(processed_self, var_len_input))
            qkv = torch.cat(embeddings, dim=1)
            attention_embedding = self.rsa(qkv, masks)
            if not input_exist:
                encoded_self = torch.cat([attention_embedding], dim=1)
                input_exist = True
            else:
                encoded_self = torch.cat([encoded_self, attention_embedding], dim=1)
        if not input_exist:
            raise UnityTrainerException(
                "The trainer was unable to process any of the provided inputs. "
                "Make sure the trained agents has at least one sensor attached to them."
            )
        return encoded_self
    def get_goal_encoding(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        encodes = []
        for idx in self._goal_processor_indices:
            processor = self.processors[idx]
            if not isinstance(processor, EntityEmbedding):
                obs_input = inputs[idx]        
                processed_obs = processor(obs_input)           
                encodes.append(processed_obs)                   
            else:
                raise UnityTrainerException(
                    "The one of the goals uses variable length observations. This use "
                    "case is not supported."
                )
        if len(encodes) != 0:
            encoded = torch.cat(encodes, dim=1)
        else:
            raise UnityTrainerException(
                "Trainer was unable to process any of the goals provided as input."
            )
        return encoded          
class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors
        total_enc_size = self.observation_encoder.total_enc_size
        total_enc_size += encoded_act_size
        if (
            self.observation_encoder.total_goal_enc_size > 0
            and network_settings.goal_conditioning_type == ConditioningType.HYPER
        ):
            self._body_endoder = ConditionalEncoder(
                total_enc_size,
                self.observation_encoder.total_goal_enc_size,
                self.h_size,
                network_settings.num_layers,
                1,
            )
        else:
            self._body_endoder = LinearEncoder(
                total_enc_size, network_settings.num_layers, self.h_size
            )
        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None                
    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)
    def copy_normalization(self, other_network: "NetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)
    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0
    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_self = self.observation_encoder(inputs)
        if actions is not None:
            encoded_self = torch.cat([encoded_self, actions], dim=1)
        if isinstance(self._body_endoder, ConditionalEncoder):
            goal = self.observation_encoder.get_goal_encoding(inputs)
            encoding = self._body_endoder(encoded_self, goal)
        else:
            encoding = self._body_endoder(encoded_self)
        if self.use_lstm:
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories
class MultiAgentNetworkBody(torch.nn.Module):
    """
    A network body that uses a self attention layer to handle state
    and action input from a potentially variable number of agents that
    share the same observation and action space.
    """
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )
        self.action_spec = action_spec
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors
        obs_only_ent_size = self.observation_encoder.total_enc_size
        q_ent_size = (
            obs_only_ent_size
            + sum(self.action_spec.discrete_branches)
            + self.action_spec.continuous_size
        )
        attention_embeding_size = self.h_size
        self.obs_encoder = EntityEmbedding(
            obs_only_ent_size, None, attention_embeding_size
        )
        self.obs_action_encoder = EntityEmbedding(
            q_ent_size, None, attention_embeding_size
        )
        self.self_attn = ResidualSelfAttention(attention_embeding_size)
        self.linear_encoder = LinearEncoder(
            attention_embeding_size,
            network_settings.num_layers,
            self.h_size,
            kernel_gain=(0.125 / self.h_size) ** 0.5,
        )
        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None                
        self._current_max_agents = torch.nn.Parameter(
            torch.as_tensor(1), requires_grad=False
        )
    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0
    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)
    def copy_normalization(self, other_network: "MultiAgentNetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)
    def _get_masks_from_nans(self, obs_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Get attention masks by grabbing an arbitrary obs across all the agents
        Since these are raw obs, the padded values are still NaN
        """
        only_first_obs = [_all_obs[0] for _all_obs in obs_tensors]
        only_first_obs_flat = torch.stack(
            [_obs.flatten(start_dim=1)[:, 0] for _obs in only_first_obs], dim=1
        )
        attn_mask = only_first_obs_flat.isnan().float()
        return attn_mask
    def _copy_and_remove_nans_from_obs(
        self, all_obs: List[List[torch.Tensor]], attention_mask: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """
        Helper function to remove NaNs from observations using an attention mask.
        """
        obs_with_no_nans = []
        for i_agent, single_agent_obs in enumerate(all_obs):
            no_nan_obs = []
            for obs in single_agent_obs:
                new_obs = obs.clone()
                new_obs[attention_mask.bool()[:, i_agent], ::] = 0.0                    
                no_nan_obs.append(new_obs)
            obs_with_no_nans.append(no_nan_obs)
        return obs_with_no_nans
    def forward(
        self,
        obs_only: List[List[torch.Tensor]],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param obs_only: Observations to be processed that do not have corresponding actions.
            These are encoded with the obs_encoder.
        :param obs: Observations to be processed that do have corresponding actions.
            After concatenation with actions, these are processed with obs_action_encoder.
        :param actions: After concatenation with obs, these are processed with obs_action_encoder.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        """
        self_attn_masks = []
        self_attn_inputs = []
        concat_f_inp = []
        if obs:
            obs_attn_mask = self._get_masks_from_nans(obs)
            obs = self._copy_and_remove_nans_from_obs(obs, obs_attn_mask)
            for inputs, action in zip(obs, actions):
                encoded = self.observation_encoder(inputs)
                cat_encodes = [
                    encoded,
                    action.to_flat(self.action_spec.discrete_branches),
                ]
                concat_f_inp.append(torch.cat(cat_encodes, dim=1))
            f_inp = torch.stack(concat_f_inp, dim=1)
            self_attn_masks.append(obs_attn_mask)
            self_attn_inputs.append(self.obs_action_encoder(None, f_inp))
        concat_encoded_obs = []
        if obs_only:
            obs_only_attn_mask = self._get_masks_from_nans(obs_only)
            obs_only = self._copy_and_remove_nans_from_obs(obs_only, obs_only_attn_mask)
            for inputs in obs_only:
                encoded = self.observation_encoder(inputs)
                concat_encoded_obs.append(encoded)
            g_inp = torch.stack(concat_encoded_obs, dim=1)
            self_attn_masks.append(obs_only_attn_mask)
            self_attn_inputs.append(self.obs_encoder(None, g_inp))
        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity, self_attn_masks)
        flipped_masks = 1 - torch.cat(self_attn_masks, dim=1)
        num_agents = torch.sum(flipped_masks, dim=1, keepdim=True)
        if torch.max(num_agents).item() > self._current_max_agents:
            self._current_max_agents = torch.nn.Parameter(
                torch.as_tensor(torch.max(num_agents).item()), requires_grad=False
            )
        num_agents = num_agents * 2.0 / self._current_max_agents - 1
        encoding = self.linear_encoder(encoded_state)
        if self.use_lstm:
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        encoding = torch.cat([encoding, num_agents], dim=1)
        return encoding, memories
class Critic(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass
    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        :param inputs: List of inputs as tensors.
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values.
        """
        pass
class ValueNetwork(nn.Module, Critic):
    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):
        nn.Module.__init__(self)
        self.network_body = NetworkBody(
            observation_specs, network_settings, encoded_act_size=encoded_act_size
        )
        if network_settings.memory is not None:
            encoding_size = network_settings.memory.memory_size // 2
        else:
            encoding_size = network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)
    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)
    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size
    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.forward(
            inputs, memories=memories, sequence_length=sequence_length
        )
        return value_outputs, critic_mem_out
    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, actions, memories, sequence_length
        )
        output = self.value_heads(encoding)
        return output, memories
class Actor(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass
    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        pass
    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        """
        Returns log_probs for actions and entropies.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param actions: AgentAction of actions.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        pass
    @abc.abstractmethod
    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """
        pass
class RewardModel:
    def __init__(self, observation_specs: List[ObservationSpec], action_spec: ActionSpec, network_settings : NetworkSettings, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=10000, activation='tanh', capacity=1e4,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 ):
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            256,
            network_settings.vis_encode_type,
            network_settings.normalize,
        )
        self.ds = self.observation_encoder.total_enc_size
        self.da = (action_spec.continuous_size
                    + sum(action_spec.discrete_branches))
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.capacity = int(capacity)
        self.buffer_seg1 = [0] * self.capacity
        self.buffer_seg2 = [0] * self.capacity
        self.buffer_label = [0] * self.capacity
        self.buffer_index = 0
        self.buffer_full = False
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*self.gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done > 0.5:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
    def clear_data(self):
        self.inputs = []
        self.targets = []       
    def add_data_batch_R(self, obses, acts, rewards, term):
        num_env = len(obses)
        for index in range(num_env):
            done = 0
            if index == num_env - 1 and term:
                done = 1
            self.add_data(obses[index], acts[index], rewards[index], done)
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])    
    def get_rank_probability(self, x_1, x_2):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    def get_entropy(self, x_1, x_2):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    def p_hat_member(self, x_1, x_2, member=-1):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        return F.softmax(r_hat, dim=-1)[:,0]
    def p_hat_entropy(self, x_1, x_2, member=-1):
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent
    def r_hat_member(self, x, member=-1):
        result_member = []
        for xx in x:
            result_member.append(self.ensemble[member](torch.from_numpy(np.asarray(xx)).float().to(device)))
        return result_member
    def r_hat_member_ori(self, x, member=-1):
        return self.ensemble[member](torch.from_numpy(x).float().to(device))
    def r_hat(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    def r_hat_batch(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member_ori(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    def get_queries(self, mb_size=20):
        for i,j in zip(self.inputs, self.targets):
            if len(i) >= self.size_segment:
                pass
            else:
                del i
                del j
        _, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        train_inputs = np.array(self.inputs[:max_len],dtype=object)
        train_targets = np.array(self.targets[:max_len],dtype=object)
        batch_index_2 = np.random.choice(max_len-1, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]                         
        r_t_2 = train_targets[batch_index_2]                
        batch_index_1 = np.random.choice(max_len-1, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]                         
        r_t_1 = train_targets[batch_index_1]                
        len_traj_1 = []
        for i in range(0, len(sa_t_1)):
            r_t_1[i] = r_t_1[i] / len(sa_t_1[i]) * 100
        len_traj_2 = []
        for i in range(0, len(sa_t_2)):
            r_t_2[i] = r_t_2[i] / len(sa_t_2[i]) * 100
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])                  
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])                  
        lenstart = 0
        time_index = []
        time_index_2 = []
        time_index_1 = []
        r_t_1_app_div = []
        return sa_t_1, sa_t_2, r_t_1, r_t_2
    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        total_sample2 = sa_t_2.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            for i in range(0, len(sa_t_1)):
                if self.buffer_index + i < self.capacity:
                    self.buffer_seg1[self.buffer_index + i] = sa_t_1[i]
            for i in range(0, len(sa_t_2)):
                if self.buffer_index + i < self.capacity:
                    self.buffer_seg2[self.buffer_index + i] = sa_t_2[i]
            self.buffer_label[self.buffer_index:self.capacity] = labels[:maximum_index]
            remain = min(total_sample - (maximum_index), total_sample2 - (maximum_index))
            if remain > 0:
                for i in range(0, remain):
                    if maximum_index + i < total_sample:
                        self.buffer_seg1[i] = sa_t_1[maximum_index + i]
                for i in range(0, remain):
                    if maximum_index + i < total_sample2:
                        self.buffer_seg2[i] = sa_t_2[maximum_index + i]
                self.buffer_label[0:remain] = labels[maximum_index:]
            self.buffer_index = remain
        else:
            for i in range(0, len(sa_t_1)):
                if self.buffer_index + i < self.capacity:
                    self.buffer_seg1[self.buffer_index + i] = sa_t_1[i]
            for i in range(0, len(sa_t_2)):
                if self.buffer_index + i < self.capacity:
                    self.buffer_seg2[self.buffer_index + i] = sa_t_2[i]
            self.buffer_label[self.buffer_index:next_index] = labels
            self.buffer_index = next_index
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_ = []
        for r in r_t_1[0]:
            sum_r_t_.append(float(sum(r)))
        sum_r_t_1 = np.array(sum_r_t_)
        sum_r_t_ = []
        for r in r_t_2[0]:
            sum_r_t_.append(float(sum(r)))
        sum_r_t_2 = np.array(sum_r_t_)
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []
            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:                               
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
        labels[margin_index] = -1 
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    def kcenter_sampling(self):
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def kcenter_disagree_sampling(self):
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def kcenter_entropy_sampling(self):
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        max_len = self.capacity if self.buffer_full else self.buffer_index
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        selected_index = self.KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def uniform_sampling(self):
        max_len = len(self.inputs)
        if max_len < 2:
            return 0
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def disagreement_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]        
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def entropy_sampling(self):
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = [self.buffer_seg1[i] for i in idxs]
                sa_t_2 = [self.buffer_seg2[i] for i in idxs]
                labels = np.array([self.buffer_label[i] for i in idxs])
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                for i in range(len(r_hat1)):
                    r_hat1[i] = r_hat1[i].sum(axis=0)
                for i in range(len(r_hat2)):
                    r_hat2[i] = r_hat2[i].sum(axis=0)
                r_hat1 = torch.stack(r_hat1)
                r_hat2 = torch.stack(r_hat2)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            loss.backward()
            self.opt.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            for member in range(self.de):
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = [self.buffer_seg1[i] for i in idxs]
                sa_t_2 = [self.buffer_seg2[i] for i in idxs]
                labels = [self.buffer_label[i] for i in idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                if member == 0:
                    total += labels.size(0)
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            loss.backward()
            self.opt.step()
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    def gen_net(self, in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(in_size, H))
            net.append(nn.LeakyReLU())
            in_size = H
        net.append(nn.Linear(in_size, out_size))
        if activation == 'tanh':
            net.append(nn.Tanh())
        elif activation == 'sig':
            net.append(nn.Sigmoid())
        else:
            net.append(nn.ReLU())
        return net
    def KCenterGreedy(self, obs, full_obs, num_new_sample):
        selected_index = []
        current_index = list(range(obs.shape[0]))
        new_obs = obs
        new_full_obs = full_obs
        start_time = time.time()
        for count in range(num_new_sample):
            dist = self.compute_smallest_dist(new_obs, new_full_obs)
            max_index = torch.argmax(dist)
            max_index = max_index.item()
            if count == 0:
                selected_index.append(max_index)
            else:
                selected_index.append(current_index[max_index])
            current_index = current_index[0:max_index] + current_index[max_index+1:]
            new_obs = obs[current_index]
            new_full_obs = np.concatenate([
                full_obs, 
                obs[selected_index]], 
                axis=0)
        return selected_index
    def compute_smallest_dist(self, obs, full_obs):
        obs = torch.from_numpy(obs).float()
        full_obs = torch.from_numpy(full_obs).float()
        batch_size = 100
        with torch.no_grad():
            total_dists = []
            for full_idx in range(len(obs) // batch_size + 1):
                full_start = full_idx * batch_size
                if full_start < len(obs):
                    full_end = (full_idx + 1) * batch_size
                    dists = []
                    for idx in range(len(full_obs) // batch_size + 1):
                        start = idx * batch_size
                        if start < len(full_obs):
                            end = (idx + 1) * batch_size
                            dist = torch.norm(
                                obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                            )
                            dists.append(dist)
                    dists = torch.cat(dists, dim=1)
                    small_dists = torch.torch.min(dists, dim=1).values
                    total_dists.append(small_dists)
            total_dists = torch.cat(total_dists)
        return total_dists.unsqueeze(1)
class SimpleActor_R(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3                                              
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size
                    + sum(self.action_spec.discrete_branches)
                ]
            ),
            requires_grad=False,
        )
        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )
        self.reward_model = RewardModel(
            observation_specs,
            action_spec,
            network_settings,
            ensemble_size=Cfg.ENSEMBLE_SIZE.value,
            lr=Cfg.REWARD_LR.value,
            mb_size=Cfg.REWARD_BATCH.value, 
            size_segment=Cfg.SEGMENT.value,
            env_maker=None, 
            max_size=10000,
            activation=Cfg.ACTIVATION.value, 
            capacity=5000,
            large_batch=Cfg.LARGE_BATCH.value, 
            label_margin=Cfg.LABEL_MARGIN.value, 
            teacher_beta=Cfg.TEACHER_BETA.value, 
            teacher_gamma=Cfg.TEACHER_GAMMA.value, 
            teacher_eps_mistake=Cfg.TEACHER_EPS_MISTAKE.value, 
            teacher_eps_skip=Cfg.TEACHER_EPS_SKIP.value,
            teacher_eps_equal = Cfg.TEACHER_EPS_EQUAL.value,
            )
    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size
    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)
    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        run_out = {}
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return action, run_out, memories
    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, actor_mem_outs = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return run_out
    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.
        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=1
        )
        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)
class SharedActorCritic(SimpleActor_R, Critic):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            observation_specs,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, self.encoding_size)
    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        return self.value_heads(encoding), memories_out
class GlobalSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.__global_step = nn.Parameter(
            torch.Tensor([0]).to(torch.int64), requires_grad=False
        )
    @property
    def current_step(self):
        return int(self.__global_step.item())
    @current_step.setter
    def current_step(self, value):
        self.__global_step[:] = value
    def increment(self, value):
        self.__global_step += value
class LearningRate(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.learning_rate = torch.Tensor([lr])
