# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-Reasoner-Zero (ORZ) is a reinforcement learning framework for training reasoning-oriented language models. It implements PPO (Proximal Policy Optimization) training using Ray for distributed computing, with models ranging from 0.5B to 32B parameters based on Qwen2.5 base models.

## Key Architecture Components

### Core Training Framework
- **PPO Training**: Main training loop in `orz/ppo/trainer.py` using Ray for distributed execution
- **Actors**: Four main actor types in `orz/ppo/actors.py`:
  - `PolicyRayActor`: Main policy model for generation
  - `RefRayActor`: Reference model for KL divergence computation
  - `CriticRayActor`: Value function estimation
  - `RewardRayActor`: Reward model for scoring responses
- **Colocation**: Actors can be colocated on same GPUs to maximize efficiency
- **VLLM Integration**: Uses VLLM engines for fast inference during training

### Experiment Configuration
- **Base Config**: `orz/exps/examples/ppo/ppo_base_exp.py` defines base PPO experiment configuration
- **Playground Scripts**: Ready-to-use training scripts in `playground/` for different model sizes
- **Resource Management**: Configurable node/GPU allocation with Ray placement groups

### Data Pipeline
- **Training Data**: Curated math datasets in `data/` (57k original + 72k extended + 13k hard)
- **Dataset Classes**: `orz/ppo/dataset.py` handles prompt formatting and data loading
- **Math Utilities**: `orz/ppo/tools/math_utils.py` for mathematical reasoning evaluation

## Development Commands

### Installation
```bash
pip install -e .
```

### Training Commands

#### Single GPU Training (0.5B model)
```bash
python -m playground.orz_0p5b_ppo_1gpu
```

#### Multi-GPU Training (0.5B model)
```bash
python -m playground.orz_0p5b_ppo
```

#### Multi-Node Training (7B model)
```bash
# On master node
ray start --head
# On worker nodes
ray start --address='<master-node-ip>:<master-node-port>'
# Start training on master node
python -m playground.orz_7b_ppo
```

#### Large Scale Training (32B model)
```bash
# 16 nodes setup
ray start --head  # master node
ray start --address='<master-node-ip>:<master-node-port>'  # worker nodes
python -m playground.orz_32b_ppo
```

### Debug Mode
```bash
DEBUG_MODE=True python -m playground.orz_14m_ppo_mini
```

### Code Quality
- **Formatting**: Uses `black` with 120 char line length
- **Linting**: Uses `flake8` with custom rules (see pyproject.toml)
- **Testing**: Uses `pytest` (configured in pyproject.toml)

## Training Configuration Patterns

### Model Scaling
- **0.5B**: Single GPU capable, good for development/debugging
- **1.5B**: 2 nodes, efficient for small-scale experiments  
- **7B**: 4 nodes, standard research scale
- **32B**: 16 nodes, production scale matching paper results

### Resource Allocation
- `colocate_all`: Run all actors on same GPUs (memory efficient)
- `colocate_critic_reward`: Share GPUs between critic and reward models
- `colocate_actor_ref`: Share GPUs between policy and reference models
- `vllm_tensor_parallel_size`: Control VLLM parallelism per engine

### Key Training Parameters
- `zero_stage`: DeepSpeed ZeRO optimization stage (typically 3)
- `adam_offload`: Offload optimizer state to CPU
- `use_compute_reward_fn`: Use mathematical reward computation
- `total_num_nodes`: Total compute nodes for distributed training

## File Structure Patterns

### Playground Scripts
Each playground script follows this pattern:
1. Import base configuration and experiment class
2. Override specific parameters for model size/resource allocation
3. Set model paths and data paths
4. Configure training hyperparameters
5. Run experiment with `asyncio.run()`

### Model Paths
- Policy models: Base Qwen2.5 models (0.5B, 1.5B, 7B, 32B)
- Critic models: Typically use same base model with value head
- Reward models: Can use separate reward-tuned models
- Checkpoints: Saved to `orz_ckpt/` directory
- Logs: TensorBoard logs in `orz_logs/`

## Docker Environment

The project includes a comprehensive Dockerfile with:
- PyTorch 24.02 base image
- Ray 2.40.0 for distributed training
- VLLM 0.6.5 for inference
- DeepSpeed 0.16.0 for optimization
- Flash Attention and other performance libraries

## Important Notes

- The codebase uses Ray for distributed coordination but colocation for memory efficiency
- Training data is mathematical reasoning focused (AIME, MATH, etc.)
- Model checkpoints and logs are stored locally in `orz_ckpt/` and `orz_logs/`
- The framework emphasizes scalability from single GPU to multi-node training
- All training scripts are designed to be runnable with minimal configuration changes