"""
Teacher–Student alternating training: teacher PPO(+KL to student) followed by
student SFT distillation on teacher-generated data.

Run (single node debug):
  DEBUG_MODE=True python -m playground.teacher_student_concurrent_distill

Multi-node (example):
  ray start --head
  # on workers: ray start --address='<master-ip>:<port>'
  python -m playground.teacher_student_concurrent_distill
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

import wandb
from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp


DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True
file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    # Core toggles
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Resources: keep conservative defaults for local testing; scale on cluster
    total_num_nodes: int = 4 if not DEBUG_MODE else 1
    actor_num: int = 2 if not DEBUG_MODE else 1

    # Colocation and parallelism
    colocate_all: bool = False
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    colocate_critic_policy: bool = True
    offload_critic_policy_colocation: bool = True

    ref_num_nodes: int = actor_num
    actor_num_nodes: int = actor_num
    critic_num_nodes: int = actor_num
    reward_num_nodes: int = actor_num
    ref_num_gpus_per_node: int = 1
    actor_num_gpus_per_node: int = 1
    critic_num_gpus_per_node: int = 1
    reward_num_gpus_per_node: int = 1

    vllm_num_engines: int = (total_num_nodes - actor_num) if not DEBUG_MODE else 1
    vllm_tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9 if not DEBUG_MODE else 0.6

    # Paths/models
    # Override these if you have local checkpoints; defaults are HF ids
    pretrain: Optional[str] = "Qwen/Qwen2.5-1.5B"
    critic_pretrain: Optional[str] = ""  # use GRPO critic head by default
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"orz_ckpt/{file_name}"
    save_path: str = ckpt_path
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # Data
    prompt_data: ListConfig = ListConfig([
        "data/strategyqa.json",
    ])
    eval_prompt_data: ListConfig = ListConfig([
        "data/eval_data/strategyqa_test.json",
        "data/eval_data/strategyqa_train.json",
    ])
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # PPO/training schedule
    train_batch_size: int = 256 if not DEBUG_MODE else 32
    num_warmup_steps: int = 5
    prompt_max_len: int = 2048
    generate_max_len: int = 2048
    max_len: int = 3072
    packing_max_len: int = generate_max_len + prompt_max_len

    n_samples_per_prompt: int = 16 if not DEBUG_MODE else 4
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 64
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    advantage_normalize: bool = False

    # Alternating schedule: few teacher rounds, then few student rounds
    initial_teacher_training_rounds: int = 2  # optional warmup of teacher-only
    teacher_training_rounds: int = 3         # teacher per cycle
    student_training_rounds: int = 3         # student per cycle

    # Teacher/student roles
    separate_teacher_model: bool = True
    # If you have a pretrained teacher, set teacher_pretrain to that path
    teacher_pretrain: Optional[str] = None
    eval_student: bool = True
    eval_teacher: bool = False
    enable_eval: bool = True if not DEBUG_MODE else True
    eval_interval: int = 10

    # Generation and augmentation for distillation
    generate_with_student: bool = False
    augment_student_generation_with_teacher: bool = True
    train_student_on_teacher_data_only: bool = True
    augment_strategy: str = "distill"  # have teacher generate for student SFT

    # Losses
    student_loss_type: str = "sft"      # distill from teacher outputs
    teacher_loss_type: str = "ppo"      # teacher optimized by PPO

    # KL shaping: penalize teacher deviations from student
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    reverse_kl: bool = True
    reward_kl_coef: float = 0.2  # KL as part of teacher reward
    kl_loss_window_size: int = 10
    kl_window_loss_coef: float = 0.01
    reward_kl_reduction: str = "mean"   # mean or sum over tokens
    kl_max_coef: float = 0.01
    kl_reward_clamp: float = 10.0

    # Misc prompting flags
    general_propmt_yes_no: bool = True
    teacher_add_role_prefix: bool = True
    teacher_explain_only: bool = False
    use_ss_reward_for_student: bool = False
    remove_student_reward_normalization: bool = True


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))

    os.makedirs(exp.cfg.save_path, exist_ok=True)
    os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    os.makedirs(exp.cfg.ckpt_path, exist_ok=True)

    run = wandb.init(
        project="open-reasoner-zero",
        name=exp.cfg.exp_name,
        sync_tensorboard=True,
        dir=exp.cfg.tensorboard_log_dir,
        config=asdict(exp.cfg),
    )

    asyncio.run(exp.run())

    run.finish()

