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

from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    # total_num_nodes: int = 16 if not DEBUG_MODE else 8
    total_num_nodes: int = 4

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "/home/a/anokhin/links/scratch/Qwen2.5-1.5B" #"/home/a/anokhin/links/scratch/Qwen2.5-1.5B-Instruct" #"/home/a/anokhin/links/scratch/Qwen2.5-1.5B" # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    e_name = 'teacherv-nonreplace-6-grpo-4gpu-v0' #"teacherv5-topr-4gpu-v0"
    exp_name: str = f"{file_name}_{e_name}"
    ckpt_path: str = f"/home/a/anokhin/links/scratch/orz_ckpt/{exp_name}"
    save_path: str = f"/home/a/anokhin/links/scratch/orz_ckpt/{exp_name}"
    tensorboard_log_dir: str = f"/home/a/anokhin/links/scratch/orz_logs/{exp_name}"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig([
        "data/strategyqa.json",
    ])
    eval_prompt_data: ListConfig = ListConfig(
        [
            # "data/eval_data/math500.json",
            # "data/eval_data/gpqa_diamond.json",
            "data/eval_data/strategyqa_test.json",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 20
    rollout_batch_size: int = 128 if not DEBUG_MODE else 128
    n_samples_per_prompt: int = 32 if not DEBUG_MODE else 8
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 240

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True if not DEBUG_MODE else False
    eval_interval: int = 10

    # generate related settings
    generate_max_len: int = 8000  # TODO: change to larger later
    max_len: int = 8192  # TODO: change to larger later
    packing_max_len: int = generate_max_len + prompt_max_len
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = True #False

    gpu_memory_utilization: float = 0.25
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0
    kl_max_coef: float = 0.01
    grpo_normalize_only_at_trainer: bool = True
    # reward_kl_coef: float = 1.0 #0.8
    # reward_match_coef: float = 1.0
    reward_kl_coef: float = 0. #1.
    reward_match_coef: float = 1. #0.1
    ss_reward_coef: float = 0. #0.33

    use_topr: bool = False
    train_teacher: bool = True
    replace_student_logprops_w_teacher: bool = False



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
