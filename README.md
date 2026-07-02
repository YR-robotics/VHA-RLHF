# VHA-RLHF: Variable-Horizon Alignment for Reinforcement Learning from Human Feedback

This repository contains the official implementation of the VHA-RLHF framework, an end-to-end architecture enabling exact temporal credit assignment across non-uniform trajectories in highly dynamic physical control environments, including UAV interception.

## System Requirements

- OS: Windows 10 (Strictly required for compatibility with the compiled Unity aerodynamic environment executable)
- GPU: NVIDIA GPU with 32GB+ VRAM
- CUDA: 12.8+
- Environment: Python 3.10+, PyTorch
## Theoretical Contributions Mapping

The primary algorithmic innovations are implemented within the customized ML-Agents core, specifically in `networks_R.py` and `on_policy_trainer_RLHF.py`.

### 1. Unsupervised State-Entropy Pre-training
(Contribution i: Resolving sparse-reward exploration stagnation)
- **Code Location**: `.\mlagents\trainers\trainer\on_policy_trainer_RLHF.py` -> `_update_policy_R(self)`
- **Mechanism**: Implemented via a phase-gated mechanism. When the training steps are below a defined threshold (`self._step <= Cfg.NUM_UNSUP_STEPS`), the algorithm bypasses standard preference updates. Instead, it calls `self.optimizer.update_state_ent()`, utilizing an intrinsic $K$-Nearest Neighbor ($K=5$) estimator to maximize state-entropy. By temporarily suspending external rewards, this forces the agent to broaden its state-space coverage and map safe operational boundaries autonomously, thereby ensuring high-quality, high-survival data for subsequent RLHF queries.

### 2. Context-Gated Trajectory Filtering Protocol
(Contribution ii: Eliminating reward distortion)
- **Code Location**: `.\mlagents\trainers\torch_entities\networks_R.py` -> `RewardModel.get_queries(self, mb_size)`
- **Mechanism**: Enforces a strict minimal-horizon threshold (`self.size_segment`, configured as `Cfg.SEGMENT.value = 20`). Before sampling preference queries, the method scans the trajectory buffer. Any rollout failing to meet the condition `if len(i) >= self.size_segment:` is actively deleted (`del i`, `del j`). This explicitly purges context-deficient "early-stall" anomalies, ensuring the surrogate reward model is strictly insulated from estimation distortion.

### 3. Native Variable-Horizon Preference Alignment (RSA-LBT)
(Contribution iii: Exact temporal credit assignment across non-uniform trajectories)
- **Code Location**: 
  - **RSA**: `.\mlagents\trainers\torch_entities\networks_R.py` -> `ObservationEncoder.forward(self)`
  - **LBT**: `.\mlagents\trainers\torch_entities\networks_R.py` -> `RewardModel.get_queries(self)` & `RewardModel.train_reward(self)`
- **Mechanism**:
  - **RSA Module**: Within the `ObservationEncoder`, a `ResidualSelfAttention` module (`self.rsa`) computes dynamic binary masks (`get_zero_entities_mask`) to natively process variable-length state-action streams, avoiding destructive artificial zero-padding.
  - **Length-Normalized BT**: The length normalization is mathematically enforced in `get_queries()` by scaling the rewards inversely to their exact unpadded sequence lengths (`r_t_1[i] = r_t_1[i] / len(sa_t_1[i])`). During optimization in `train_reward()`, the logits are dynamically summed across the valid temporal dimension (`r_hat1[i].sum(axis=0)`) and optimized via Cross-Entropy, executing an exact length-normalized Bradley-Terry objective:
    $$P_\psi(\sigma^1 \succ \sigma^2) = \frac{\exp \left( \frac{1}{L_1} \sum_{t \in \sigma^1} r_\psi(s_t, a_t) \right)}{\sum_{m \in \{1,2\}} \exp \left( \frac{1}{L_m} \sum_{t \in \sigma^m} r_\psi(s_t, a_t) \right)}$$


## Environment Setup
We recommend using Anaconda to manage your Python environment. To create a new environment and install the required dependencies, execute the following commands in your terminal:
```bash
conda create -n vha_rlhf python=3.10 -y
conda activate vha_rlhf
pip install -r requirements.txt
```
## Training the Agent
To execute the training pipeline in the UAV interception simulator, execute the following command from the project root:

```bash
python learn.py .\trainer_config_vha.yaml --run-id=your_experiment_id --force --env=AB_multi_release_v1.0_windows_seed\air_combat_URP.exe
```
## Monitoring Training Progress
To monitor the training dynamics in real-time, including the reward function curves and episodic success rates, you can utilize TensorBoard. Open a new terminal window, ensure your conda environment is activated, and execute:
```bash
tensorboard --logdir=./results
```
