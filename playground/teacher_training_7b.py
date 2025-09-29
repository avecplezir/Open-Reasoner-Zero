"""
Qwen2.5-1.5B base model + ppo


running command in 2 nodes:

on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`
then on master node, run `python -m playground.orz_1p5b_ppo`


debug running command in 1 nodes:
run `DEBUG_MODE=True python -m playground.orz_1p5b_ppo`

"""


import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import wandb, omegaconf
from dataclasses import asdict
import random

from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)

prefix = '/home/a/anokhin/links/scratch'
# prefix = '/home/anokhin/scratch'

@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    # total_num_nodes: int = 16 if not DEBUG_MODE else 8
    total_num_nodes: int = 4

    actor_num = 3
    # resource related settings
    ref_num_nodes: int = actor_num
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = actor_num
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = actor_num
    critic_num_gpus_per_node: int = 1
    reward_num_nodes: int = actor_num
    reward_num_gpus_per_node: int = 1
    colocate_all: bool = False
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    colocate_critic_policy: bool = True
    offload_critic_policy_colocation: bool = True
    vllm_num_engines: int = total_num_nodes - actor_num
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3
    vllm_sync_backend: str = "gloo"  # nccl or gloo

    # path related settings
    pretrain: Optional[str] = f"{prefix}/binary_noncol_orz_7b_sft_7B-student-data-v0-143/iter20/policy" #f"{prefix}/binary_noncol_orz_7b_ppo_7B-student-data-v0-759/iter50/policy"  #f"{prefix}/binary_noncol_orz_1p5b_ppo_grpo-base-explain-v0-824/iter150/policy" #f"{prefix}/iter104/policy" #f"{prefix}/iter50/policy" #f"{prefix}/Qwen2.5-1.5B" # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 10
    # current date and time
    randint = random.randint(0, 1000)
    e_name = f'7b-iter20-correct-v0-{randint}'
    exp_name: str = f"{file_name}_{e_name}"
    ckpt_path: str = f"{prefix}/orz_ckpt/{exp_name}"
    save_path: str = f"{prefix}/orz_ckpt/{exp_name}"
    tensorboard_log_dir: str = f"{prefix}/orz_logs/{exp_name}"

    # data related settings
    prompt_data: ListConfig = ListConfig([
        "data/strategyqa.json",
    ])
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/strategyqa_test.json",
            "data/eval_data/strategyqa_train.json",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    num_warmup_steps: int = 5
    prompt_max_len: int = 2048

    advantage_normalize: bool = False

    num_episodes: int = 7
    n_samples_per_prompt: int = 16 if not DEBUG_MODE else 4

    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0

    enable_eval: bool = True if not DEBUG_MODE else False
    eval_interval: int = 10

    # generate related settings
    generate_max_len: int = 2048 #12000 #8000  # 2000 #4000 # TODO: change to larger later
    max_len: int = 3072 #12192 #8192  #2560 #4192 # TODO: change to larger later
    packing_max_len: int = generate_max_len + prompt_max_len

    # grpo related settings
    use_grpo: bool = True #False

    gpu_memory_utilization: float = 0.95
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    initial_teacher_training_rounds: int = 0
    student_training_rounds: int = 1  # number student training rounds, -1 means no student training
    teacher_training_rounds: int = 1  # number teacher training rounds, -1 means no teacher training
    student_teacher_order: bool = True

    generate_with_student: bool = True
    augment_student_generation_with_teacher: bool = True
    augment_strategy: str = "correct"  # options: correct | yes_no | only_wrong | opposite

    separate_teacher_model: bool = False
    teacher_pretrain: Optional[str] = pretrain

    train_teacher_on_teacher_data_only: bool = True
    train_teacher_on_student_data_only: bool = False

    skip_student_training_to_debug: bool = True
    skip_student_first_n_rounds: int = 0


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
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
