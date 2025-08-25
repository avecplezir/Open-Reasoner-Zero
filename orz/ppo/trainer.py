import asyncio
import json
import math
import os
import random
from functools import partial
from heapq import heapify, heappop, heappush
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union
import wandb
import numpy as np
from collections import defaultdict
from deepspeed.accelerator import get_accelerator

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from orz.ppo.actors import PPORayActorGroup
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
    normalize_advantages,
)

from playground.zero_setting_base import create_teacher_prompt_from_answer


class RayPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        strategy: DeepspeedStrategy,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        vllm_engines=None,
        colocate_pg: Optional[PlacementGroup] = None,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.vllm_engines = vllm_engines
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg

        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)
        self.student_replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,
            cpu_offload=True,
            packing_samples=True,
        )
        self.teacher_replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,
            cpu_offload=True,
            packing_samples=True,
        )

    def __del__(self):
        self.writer.close()

    async def eval(self):
        raise NotImplementedError("Eval function should be implemented in user's exp")

    async def train(self):
        # 1. create rank0 policy model and vllm_engines groups, then boardcast weights to vllm engins
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()
            await self._backload_vllm_engines()

        await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)

        # Initialize teacher model's own process group with same vLLM engines if separate teacher is enabled
        if self.cfg.separate_teacher_model:
            async with Timer("teacher init vllm engines actor group"):
                await self.teacher_model.async_run_method("_init_teacher_vllm_engines_actor_group", self.vllm_engines)
        
        logger.info("Create vllm engine gourps done.")

        async with Timer("Sync actor weights to vllm engines"):
            await self._sync_policy_weights_to_vllm()

        if self.cfg.colocate_all:
            async with Timer("Offload policy model to cpu"):
                await self.policy_model.offload_to_cpu()

        # 2. main training loop
        consumed_samples = 0
        num_rollouts_per_episodes = (
            self.num_update_steps_per_episodes
            * self.cfg.train_batch_size
            // self.cfg.max_epochs
            // self.cfg.rollout_batch_size
            // self.cfg.n_samples_per_prompt
        )

        self.global_step = consumed_samples // self.cfg.rollout_batch_size
        start_episode = consumed_samples // self.cfg.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * self.cfg.rollout_batch_size)

        for episode in range(start_episode, self.cfg.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()), desc=f"Episode [{episode + 1}/{self.cfg.num_episodes}]"
            )
            for iter, rand_prompts in enumerate(self.prompts_dataloader):

                # 1. eval if enable eval
                if self.cfg.enable_eval and (
                    self.global_step % self.cfg.eval_interval == 0 or iter == len(self.prompts_dataloader) - 1
                ):
                    await self.eval()

                # 3. make experiences, calculate advantages and returns
                await self.make_experience(rand_prompts)

                # check if has enough data
                if len(self.student_replay_buffer) <= 0 or len(self.teacher_replay_buffer) <= 0:
                    if self.cfg.colocate_all:
                        # skip, but transfer weight
                        await self.policy_model.backload_to_gpu()
                        await self._backload_vllm_engines()
                        await self._sync_policy_weights_to_vllm()
                        await self.policy_model.offload_to_cpu()
                        if self.cfg.separate_teacher_model:
                            await self.teacher_model.offload_to_cpu()
                    continue

                if self.cfg.advantage_normalize:
                    self.student_replay_buffer = normalize_advantages(self.student_replay_buffer)
                    self.teacher_replay_buffer = normalize_advantages(self.teacher_replay_buffer)

                if self.cfg.student_training_frequency > 0:
                    if (episode + 1) % self.cfg.student_training_frequency == 0:
                        logger.info(f'training student model, {episode + 1} episode')
                        train_set = zip([self.student_replay_buffer], ["",])
                        self.teacher_replay_buffer.clear()
                    else:
                        logger.info(f'training teacher model, {episode + 1} episode')
                        train_set = zip([self.teacher_replay_buffer], ["teacher",])
                        self.student_replay_buffer.clear()
                else:
                    if self.cfg.student_teacher_order:
                        train_set = zip([self.student_replay_buffer, self.teacher_replay_buffer], ["", 'teacher'])
                    else:
                        train_set = zip([self.teacher_replay_buffer, self.student_replay_buffer], ['teacher', ''])

                for replay_buffer, prefix in train_set:
                    model = self.teacher_model if self.cfg.separate_teacher_model and prefix == "teacher" else self.policy_model

                    logger.info(f"Start training {prefix} model, replay buffer size: {len(replay_buffer)}")
                    # serialize replay buffer to jsonl
                    async with Timer("Dumping replay buffer"):
                        all_replay_buffer_save_path = os.path.join(self.cfg.save_path, "dumped_replay_buffer")
                        os.makedirs(all_replay_buffer_save_path, exist_ok=True)
                        dump_path = os.path.join(all_replay_buffer_save_path,
                                                 f"iter{self.global_step}_{prefix}_replay_buffer.jsonl")
                        with open(dump_path, "a") as f:
                            logger.info(f"dumping replay buffer to {dump_path}")
                            for item in replay_buffer:
                                f.write(json.dumps(item.to_json()) + "\n")

                    num_policy_dp_nodes = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
                    num_critic_dp_nodes = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
                    policy_buffers = replay_buffer.split_to_n_batches(num_policy_dp_nodes)
                    if num_policy_dp_nodes != num_critic_dp_nodes:
                        critic_buffers = replay_buffer.split_to_n_batches(num_critic_dp_nodes)
                    else:
                        critic_buffers = policy_buffers

                    # 4. train policy/critic model
                    if self.cfg.colocate_all:
                        if self.critic_model is not None:
                            async with Timer("Critic model training"):
                                await self.critic_model.backload_to_gpu()
                                await self.ppo_local_train_critic(critic_buffers, self.global_step, prefix)
                                await self.critic_model.offload_to_cpu()
                        async with Timer("Actor model training"):
                            await model.backload_to_gpu()
                            status = await self.ppo_local_train_policy(policy_buffers, self.global_step, prefix)
                            await model.offload_to_cpu()

                    else:
                        if self.critic_model is not None:
                            async with Timer("Actor and Critic model training"):
                                status = await asyncio.gather(
                                    self.ppo_local_train_policy(policy_buffers, self.global_step, prefix),
                                    self.ppo_local_train_critic(critic_buffers, self.global_step, prefix),
                                )
                                await asyncio.gather(
                                    model.async_run_method("empty_cache"),
                                    self.critic_model.async_run_method("empty_cache"),
                                )
                                status = status[0]
                        else:
                            async with Timer("Actor model training"):
                                status = await self.ppo_local_train_policy(policy_buffers, self.global_step, prefix)
                                await model.async_run_method("empty_cache")

                    replay_buffer.clear()

                    # 5. set logs
                    logger.info(f'{prefix} {status}')

                pbar.update()
                # log epoch info
                self.writer.add_scalar("episode_idx", episode, self.global_step)
                self.global_step += 1
                if self.global_step % self.cfg.save_interval == 0:
                    await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                    if self.critic_model is not None:
                        await self.critic_model.async_save_model(self.tokenizer, self.global_step)
                    logger.info("Successfully save model weights, training continue.")

                if self.cfg.separate_teacher_model and self.cfg.update_teacher_freq > 0 and \
                        self.global_step % self.cfg.update_teacher_freq == 0:
                    # update teacher model with policy model
                    # logger.info(f"Saving current policy model at step {self.global_step}")
                    # await self.policy_model.backload_to_gpu()
                    # await self.policy_model.async_save_model(self.tokenizer, '_current')
                    # await self.policy_model.offload_to_cpu()

                    # logger.info(f"Update teacher model with policy model at step {self.global_step}")
                    # await self.teacher_model.backload_to_gpu()
                    # await self.teacher_model.async_load_checkpoint(self.strategy, '/home/a/anokhin/links/scratch/iter104') #os.path.join(self.cfg.save_path, f"iter_current", "policy"))
                    # await self.teacher_model.offload_to_cpu()
                    # logger.info("Successfully update teacher model with policy model, training continue.")
                    logger.info(f"Exporting policy params {self.global_step}")
                    ref = await self.policy.async_export_params()
                    logger.info(f"Loading policy params to teacher {self.global_step}")
                    await self.teacher_model.async_load_params(ref)

                if self.cfg.colocate_all:
                    async with Timer("Backload vllm engines to gpu and sync policy weights after training"):
                        await self.policy_model.backload_to_gpu()
                        await self._backload_vllm_engines()
                        await self._sync_policy_weights_to_vllm()
                        await self.policy_model.offload_to_cpu()

            if self.cfg.update_ref_every_epoch:
                await self.policy_model.backload_to_gpu()
                await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                await self.policy_model.offload_to_cpu()
                await asyncio.gather(
                    *self.ref_model.async_init_model_from_pretrained(
                        self.strategy, os.path.join(self.cfg.save_path, f"iter{self.global_step}", "policy")
                    )
                )
                logger.info("Successfully update ref model with policy model, training continue.")


        await self.policy_model.async_save_model(self.tokenizer, self.cfg.num_episodes * len(self.prompts_dataloader))
        if self.cfg.separate_teacher_model:
            await self.teacher_model.async_save_model(self.tokenizer, f'teacher-{self.cfg.num_episodes * len(self.prompts_dataloader)}')

        logger.info("Successfully save model weights, training done.")


    @torch.no_grad()
    async def make_experience(self, all_inputs: Union[Tuple[str, dict], List[Tuple[str, dict]]], **generate_kwargs):
        teacher_experiences = []
        student_experiences = []

        combined_all_student_prompts, combined_all_teacher_prompts, combined_outputs, combined_custom_rewards, combined_teacher_custom_rewards, combined_answer_indices, combined_initial_scores, combined_initial_teacher_scores, combined_final_answers = [], [], [], [], [], [], [], [], []
        teacher_generated = []

        if self.cfg.generate_with_student:
            # the same, but now generate data with student prompts
            # Create paired data (positive/negative for each prompt)
            paired_data = []
            for prompt in all_inputs:
                for _ in range(2*self.cfg.n_samples_per_prompt):
                    # Create a pair: (student, teacher_yes, teacher_no, extra)
                    paired_data.append((
                        prompt[0],  # student prompt
                        prompt[1]  # extra info
                    ))

            # Shuffle the pairs to randomize order, but keep pairs together
            rng = random.Random(42)
            rng.shuffle(paired_data)

            # Flatten into separate lists, ensuring each pair stays together
            all_student_prompts = []
            all_extras = []
            for student, extra in paired_data:
                # Add both positive and negative examples
                all_student_prompts.extend([student])
                all_extras.extend([extra])

            # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence
            # 1.1 generate sequences via vllm engines
            outputs = []
            num_vllm_dp_gruops = len(self.vllm_engines)

            # Sync student (policy) model weights to VLLM engines before generation
            async with Timer("Sync policy weights to VLLM engines for student generation"):
                # await self._sync_teacher_weights_to_vllm()
                await self.policy_model.backload_to_gpu()
                await self._sync_policy_weights_to_vllm()
                await self.policy_model.offload_to_cpu()

            async with Timer("Generate student sequences via vllm engines"):
                dp_prompt_size = (len(all_student_prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops
                dp_tasks = []
                for dp_rank in range(num_vllm_dp_gruops):
                    # Use teacher prompts for generation (they have the ground truth answer)
                    dp_student_inputs = all_student_prompts[dp_rank * dp_prompt_size: (dp_rank + 1) * dp_prompt_size]
                    dp_extras = all_extras[dp_rank * dp_prompt_size: (dp_rank + 1) * dp_prompt_size]
                    # handle last batch has no enough data
                    if len(dp_student_inputs) <= 0:
                        continue
                    gen_func = self._get_generate_function(dp_rank)
                    dp_tasks.append(self.generate_vllm(gen_func, dp_student_inputs, extras=dp_extras, **generate_kwargs))

                logger.info("start generation from student prompts")
                local_responses = await asyncio.gather(*dp_tasks)
                outputs.extend(sum(local_responses, []))
                logger.info("generate local rollout batch done")

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_student_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

            # 1.2 calculate custom rewards if has custom reward function
            if self.cfg.use_compute_reward_fn:
                async with Timer("Calculate custom rewards"):
                    dp_tasks = []
                    reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                    # Use student prompts for reward calculation since that's what the model will be trained on
                    all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers = await reward_fn(
                        all_student_prompts, outputs, all_extras)
                    assert len(all_student_prompts) == len(outputs), "generate objects number after custom reward function must be equal to all inputs number"
            else:
                all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers = all_student_prompts, outputs, None, None, None, None, None, None

            # create teacher prompts from student prompts
            if self.tokenizer.bos_token_id is None:
                bos_token = ""
            else:
                bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

            all_teacher_prompts = []
            for all_extra, final_answer, score in zip(all_extras, final_answers, initial_teacher_scores):

                if score:
                    # logger.info(f"score {score}, final_answer {final_answer}")
                    teacher_prompt = create_teacher_prompt_from_answer(all_extra["dialogue"], final_answer, bos_token)
                else:
                    if random.random() > 0.5:
                        teacher_prompt = all_extra["teacher_prompt_yes"]
                    else:
                        teacher_prompt = all_extra["teacher_prompt_no"]

                all_teacher_prompts.append(teacher_prompt)

            assert len(all_student_prompts) == len(all_teacher_prompts), "student and teacher prompts must be equal in length"

            teacher_generated.extend([False] * len(all_student_prompts))

            combined_all_student_prompts.extend(all_student_prompts)
            combined_all_teacher_prompts.extend(all_teacher_prompts)
            combined_outputs.extend(outputs)
            combined_custom_rewards.extend(custom_rewards)
            combined_teacher_custom_rewards.extend(teacher_custom_rewards)
            combined_answer_indices.extend(answer_indices)
            combined_initial_scores.extend(initial_scores)
            combined_initial_teacher_scores.extend(initial_teacher_scores)
            combined_final_answers.extend(final_answers)

        if self.cfg.augment_student_generation_with_teacher and self.cfg.generate_with_student:

            # Sync teacher model weights to VLLM engines before generation
            if self.cfg.separate_teacher_model:
                async with Timer("Sync teacher weights to VLLM engines"):
                    await self.teacher_model.backload_to_gpu()
                    await self._sync_teacher_weights_to_vllm()
                    await self.teacher_model.offload_to_cpu()

            # create the oposite teacher prompt and collect data with it
            all_teacher_prompts = []
            aug_all_student_prompts = []
            aug_all_extras = []
            logger.info(f"initial_teacher_scores {len(initial_teacher_scores)}, all_extras {len(all_extras)} all_student_prompts {len(all_student_prompts)}")
            for i, (teacher_score, extra, student_prompt) in enumerate(zip(initial_teacher_scores, all_extras, all_student_prompts)):
                if teacher_score:
                    if 'yes' in final_answer.lower():
                        opposite_answer = 'no'
                    elif 'no' in final_answer.lower():
                        opposite_answer = 'yes'
                    else:
                        assert False, f"final_answer {final_answer} must be yes or no"

                    teacher_prompt = create_teacher_prompt_from_answer(extra["dialogue"], opposite_answer, bos_token)
                    # logger.info(f"teacher_score {teacher_score}, final_answer {final_answer}, opposite_answer {opposite_answer}")
                    # logger.info(f"teacher_prompt {teacher_prompt} \n, student_prompt {student_prompt}")
                    extra["teacher_answer"] = opposite_answer

                    all_teacher_prompts.append(teacher_prompt)
                    aug_all_student_prompts.append(student_prompt)
                    aug_all_extras.append(extra)

                # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence
                # 1.1 generate sequences via vllm engines
            all_extras = aug_all_extras
            all_student_prompts = aug_all_student_prompts
            outputs = []
            num_vllm_dp_gruops = len(self.vllm_engines)

            async with Timer("Generate complimentary teacher sequences via vllm engines"):
                dp_prompt_size = (len(all_teacher_prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops
                dp_tasks = []
                for dp_rank in range(num_vllm_dp_gruops):
                    # Use teacher prompts for generation (they have the ground truth answer)
                    dp_teacher_inputs = all_teacher_prompts[dp_rank * dp_prompt_size: (dp_rank + 1) * dp_prompt_size]
                    dp_extras = all_extras[dp_rank * dp_prompt_size: (dp_rank + 1) * dp_prompt_size]
                    # handle last batch has no enough data
                    if len(dp_teacher_inputs) <= 0:
                        continue
                    gen_func = self._get_generate_function(dp_rank)
                    dp_tasks.append(
                        self.generate_vllm(gen_func, dp_teacher_inputs, extras=dp_extras, **generate_kwargs))

                logger.info("start generation from complimentary teacher prompts")
                local_responses = await asyncio.gather(*dp_tasks)
                outputs.extend(sum(local_responses, []))
                logger.info("generate local rollout batch done")

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_teacher_prompts) == len(
                outputs), "generate objects number must be equal to all inputs number"

            # 1.2 calculate custom rewards if has custom reward function
            if self.cfg.use_compute_reward_fn:
                async with Timer("Calculate custom rewards"):
                    dp_tasks = []
                    reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                    # Use student prompts for reward calculation since that's what the model will be trained on
                    all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers = await reward_fn(
                        all_student_prompts, outputs, all_extras)
                    assert len(all_student_prompts) == len(outputs) == len(
                        all_teacher_prompts), "generate objects number after custom reward function must be equal to all inputs number"
            else:
                all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers = all_student_prompts, outputs, None, None, None, None, None, None

            teacher_generated.extend([True] * len(all_student_prompts))

            combined_all_student_prompts.extend(all_student_prompts)
            combined_all_teacher_prompts.extend(all_teacher_prompts)
            combined_outputs.extend(outputs)
            combined_custom_rewards.extend(custom_rewards)
            combined_teacher_custom_rewards.extend(teacher_custom_rewards)
            combined_answer_indices.extend(answer_indices)
            combined_initial_scores.extend(initial_scores)
            combined_initial_teacher_scores.extend(initial_teacher_scores)
            combined_final_answers.extend(final_answers)

        # offload vllm engines when colocate all models
        if self.cfg.colocate_all:
            async with Timer("Offload vllm engines to cpu"):
                await self._offload_vllm_engines()

        all_student_prompts, all_teacher_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers = \
            combined_all_student_prompts, combined_all_teacher_prompts, combined_outputs, combined_custom_rewards, combined_teacher_custom_rewards, combined_answer_indices, combined_initial_scores, combined_initial_teacher_scores, combined_final_answers

        # Randomize order of all arrays
        indices = np.random.permutation(len(all_student_prompts))
        all_student_prompts = [all_student_prompts[i] for i in indices]
        all_teacher_prompts = [all_teacher_prompts[i] for i in indices]
        outputs = [outputs[i] for i in indices]
        custom_rewards = [custom_rewards[i] for i in indices]
        teacher_custom_rewards = [teacher_custom_rewards[i] for i in indices]
        answer_indices = [answer_indices[i] for i in indices]
        initial_scores = [initial_scores[i] for i in indices]
        initial_teacher_scores = [initial_teacher_scores[i] for i in indices]
        final_answers = [final_answers[i] for i in indices]
        teacher_generated = [teacher_generated[i] for i in indices]

        initial_scores, initial_teacher_scores = np.array(initial_scores), np.array(initial_teacher_scores)

        logger.info(f"all_student_prompts: {len(all_student_prompts)}, all_teacher_prompts: {len(all_teacher_prompts)}")
        assert len(all_student_prompts) == len(all_teacher_prompts), logger.info(f"student and teacher prompts must be equal in length {len(all_student_prompts)} {len(all_teacher_prompts)}")

        ic = np.logical_and(initial_scores == 0, initial_teacher_scores == 1)
        cc = np.logical_and(initial_scores == 1, initial_teacher_scores == 1)
        ii = np.logical_and(initial_scores == 0, initial_teacher_scores == 0)

        # empty data
        if len(all_student_prompts) == 0:
            return

        # 1.3 packing samples
        async with Timer("Packing samples"):
            # Pack student sequences (for training)
            (
                ret_sequences,
                ret_attention_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
                ret_teacher_sequences,
                ret_teacher_attention_masks,
                ret_teacher_num_actions,
                ret_teacher_packed_seq_lens,
                ret_teacher_custom_rewards,
            ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                all_student_prompts, all_teacher_prompts, outputs, custom_rewards, teacher_custom_rewards, self.cfg.packing_max_len,
            )
            action_masks = None
            teacher_action_masks = None

        # 1.4 inference and calculate values, log probs, rewards, kl divergence for student sequences
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence for student"):
            student_experiences = await self.inference_and_calculates(
                ret_sequences,
                ret_attention_masks,
                action_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            )
            logger.info(f"student experiences size: {len(student_experiences)} {len(ret_num_actions)}")

        # 1.5 inference and calculate values, log probs, rewards, kl divergence for teacher sequences
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence for teacher"):
            teacher_experiences = await self.inference_and_calculates(
                ret_teacher_sequences,
                ret_teacher_attention_masks,
                teacher_action_masks,
                ret_teacher_num_actions,
                ret_teacher_packed_seq_lens,
                ret_teacher_custom_rewards,
                use_teacher_model=self.cfg.separate_teacher_model,
            )
            logger.info(f"teacher experiences size: {len(teacher_experiences)} {len(ret_teacher_num_actions)}")

        # Compute teacher reward
        async with Timer("Calculate teacher reward, replace student or teacher log probs w/ the correct one, compute torp ratio"):
            final_reward_list = []
            kl_max_list = []
            kl_mean_list = []
            kl_sum_list = []
            match_reward_list = []
            ss_reward_mean_list = []
            ss_reward_min_list = []
            ss_reward_list = []
            teacher_ratio_clipped_0_1_list = []
            student_ratio_clipped_0_1_list = []

            teacher_prompt_idx = 0
            teacher_pass_at_n_dict = defaultdict(list)
            for student_exp, teacher_exp in zip(student_experiences, teacher_experiences):

                kl_div_all = compute_approx_kl(
                    teacher_exp.action_log_probs,
                    student_exp.action_log_probs if not self.cfg.reward_kl_toward_ref_model else student_exp.base_action_log_probs,
                    action_mask=student_exp.action_mask,
                    use_kl_estimator_k3=self.cfg.use_kl_estimator_k3,
                    use_abs_kl=self.cfg.use_abs_kl,
                )

                offset = 0
                seq_offset = 0
                total_lengths = student_exp.info["total_length"].flatten()
                for i, num_action in enumerate(teacher_exp.num_actions[0]):
                    na = int(num_action.item())
                    seq_len = int(total_lengths[i])
                    prompt_len = seq_len - na

                    # computing answer alignment reward
                    final_answer_start, final_answer_end = answer_indices[teacher_prompt_idx]
                    if final_answer_start is not None and final_answer_start < final_answer_end:

                        final_answer_start, final_answer_end = offset + final_answer_start, offset + final_answer_end
                        final_answer_log_propbs = student_exp.action_log_probs[:, final_answer_start:final_answer_end]
                        # logger.info(f'student_exp.action_log_probs: {student_exp.action_log_probs.shape} {final_answer_start} {final_answer_end} {s_final_answer_start} {s_final_answer_end}')
                        # s_final_answer_start, s_final_answer_end = seq_offset + prompt_len + final_answer_start, seq_offset + prompt_len + final_answer_end
                        # s_final_answer_log_propbs = student_exp.action_log_probs[:, s_final_answer_start:s_final_answer_end]
                        # logger.info(f'final_answer_log_propbs: {final_answer_log_propbs.shape}')
                        # check if we find indices correctly
                        # vis_final_answer = self._detokenize(student_exp.sequences[0][s_final_answer_start:s_final_answer_end])
                        # logger.info(f"start end: {s_final_answer_start, s_final_answer_end}, vis_final_answer: {vis_final_answer} final_answer_log_propbs {final_answer_log_propbs}")
                        ss_reward_mean = final_answer_log_propbs.mean().item()
                        ss_reward_min = final_answer_log_propbs.min().item()
                        ss_reward = self.cfg.kl_mean_coef * ss_reward_mean + self.cfg.kl_max_coef * ss_reward_min
                    else:
                        ss_reward_mean = ss_reward_min = ss_reward = -1.1

                    ss_reward_mean_list.append(ss_reward_mean)
                    ss_reward_min_list.append(ss_reward_min)
                    ss_reward_list.append(ss_reward)

                    # computing kl divergence reward
                    start, end = offset, offset + na
                    kl_episode = kl_div_all[:, start:end].clone()
                    kl_max = torch.max(kl_episode.abs(), dim=-1)[0]
                    kl_mean = masked_mean(kl_episode, None, dim=-1)
                    kl_sum = kl_episode.sum(dim=-1)
                    if self.cfg.reward_kl_reduction == "mean":
                        kl_reward = -self.cfg.kl_mean_coef * kl_mean - self.cfg.kl_max_coef * kl_max
                    elif self.cfg.reward_kl_reduction == "sum":
                        kl_reward = -self.cfg.kl_mean_coef * kl_sum - self.cfg.kl_max_coef * kl_max
                    kl_reward = torch.clamp(kl_reward, min=-self.cfg.kl_reward_clamp)

                    match_reward_check = teacher_custom_rewards[teacher_prompt_idx][-1]
                    match_reward = teacher_exp.info['custom_rewards'][i][-1]
                    assert match_reward_check == match_reward, "match_reward_check and match_reward must be equal"
                    final_reward_list.append(self.cfg.ss_reward_coef * ss_reward_list[-1] + self.cfg.reward_kl_coef * kl_reward + self.cfg.reward_match_coef * match_reward)
                    teacher_pass_at_n_dict[all_teacher_prompts[teacher_prompt_idx]].append(final_reward_list[-1].item())

                    kl_max_list.append(kl_max.item())
                    kl_mean_list.append(kl_mean.item())
                    kl_sum_list.append(kl_sum.item())
                    match_reward_list.append(match_reward.item())

                    # compute ratio_clipped_0_1 for TOPR
                    if self.cfg.use_topr:
                        if teacher_generated[teacher_prompt_idx]:
                            teacher_ratio_clipped_0_1_scalar = torch.tensor(1)
                            student_ratio_clipped_0_1_scalar = torch.exp((student_exp.action_log_probs[:, start:end].sum(-1) - teacher_exp.action_log_probs[:, start:end].sum(-1)).clamp(max=0.0))
                            student_exp.ratio_clipped_0_1[:, start:end] = student_ratio_clipped_0_1_scalar
                        else:
                            student_ratio_clipped_0_1_scalar = torch.tensor(1)
                            teacher_ratio_clipped_0_1_scalar = torch.exp((teacher_exp.action_log_probs[:, start:end].sum(-1) - student_exp.action_log_probs[:, start:end].sum(-1)).clamp(max=0.0))
                            teacher_exp.ratio_clipped_0_1[:, start:end] = teacher_ratio_clipped_0_1_scalar

                        student_exp.info['use_topr'] = torch.tensor(1.).unsqueeze(0)
                        teacher_exp.info['use_topr'] = torch.tensor(1.).unsqueeze(0)
                        teacher_ratio_clipped_0_1_list.append(teacher_ratio_clipped_0_1_scalar.item())
                        student_ratio_clipped_0_1_list.append(student_ratio_clipped_0_1_scalar.item())

                    if not teacher_generated[i]:
                        if self.cfg.replace_teacher_logprops_w_student:
                            teacher_exp.action_log_probs[:, start:end] = student_exp.action_log_probs[:, start:end]
                        if self.cfg.replace_teacher_base_logprops_w_student and not teacher_generated[i]:
                            teacher_exp.base_action_log_probs[:, start:end] = student_exp.base_action_log_probs[:, start:end]
                    else:
                        if self.cfg.replace_student_logprops_w_teacher:
                            student_exp.action_log_probs[:, start:end] = teacher_exp.action_log_probs[:, start:end]
                        if self.cfg.replace_student_base_logprops_w_teacher:
                            student_exp.base_action_log_probs[:, start:end] = teacher_exp.base_action_log_probs[:, start:end]

                    offset += na
                    teacher_prompt_idx += 1
                    seq_offset += seq_len

                    if not self.cfg.use_grpo:
                        teacher_exp.info['custom_rewards'][i][-1] = final_reward_list[-1]

                assert kl_div_all.shape[1] == end, "number of action should be the same in kl and num_actions"
                assert len(student_exp.sequences[0]) == seq_offset, "student_exp.sequences must be equal to seq_offset at the end"

            assert len(final_reward_list) == teacher_prompt_idx == len(all_teacher_prompts), "kl_reward_list and last teacher prompt idx and all_teacher_prompts must be equal to all teacher prompts length"

            # Log average KL divergence between student and teacher
            kl_mean_list = np.array(kl_mean_list)
            kl_sum_list = np.array(kl_sum_list)
            kl_max_list = np.array(kl_max_list)
            ss_reward_mean_list = np.array(ss_reward_mean_list)
            ss_reward_min_list = np.array(ss_reward_min_list)
            match_reward_list = np.array(match_reward_list)
            teacher_ratio_clipped_0_1_list = np.array(teacher_ratio_clipped_0_1_list)
            student_ratio_clipped_0_1_list = np.array(student_ratio_clipped_0_1_list)
            avg_student_teacher_kl = sum(kl_mean_list) / len(kl_mean_list)
            avg_student_teacher_kl_max = sum(kl_max_list) / len(kl_max_list)
            avg_match_reward = sum(match_reward_list) / len(match_reward_list) if len(match_reward_list) > 0 else 0

            correct_match_reward_trainer = np.array([]) if np.all(initial_scores == 0) else np.array(match_reward_list[initial_scores == 1])
            incorrect_match_reward_trainer = np.array([]) if np.all(initial_scores == 1) else np.array(match_reward_list[initial_scores == 0])
            correct_kl_sum_list = np.array([]) if np.all(ic) else np.array(kl_sum_list[cc])
            incorrect_kl_sum_list = np.array([]) if np.all(cc) else np.array(kl_sum_list[ic])
            correct_kl_mean_list = np.array([]) if np.all(ic) else np.array(kl_mean_list[cc])
            incorrect_kl_mean_list = np.array([]) if np.all(cc) else np.array(kl_mean_list[ic])
            correct_kl_max_list = np.array([]) if np.all(ic) else np.array(kl_max_list[cc])
            incorrect_kl_max_list = np.array([]) if np.all(cc) else np.array(kl_max_list[ic])
            correct_ss_reward_mean_list = np.array([]) if np.all(ic) else np.array(ss_reward_mean_list[cc])
            incorrect_ss_reward_mean_list = np.array([]) if np.all(cc) else np.array(ss_reward_mean_list[ic])
            correct_ss_reward_min_list = np.array([]) if np.all(ic) else np.array(ss_reward_min_list[cc])
            incorrect_ss_reward_min_list = np.array([]) if np.all(cc) else np.array(ss_reward_min_list[ic])
            teacher_correct_ratio_clipped_0_1_list = np.array([]) if np.all(ic) or not self.cfg.use_topr else np.array(teacher_ratio_clipped_0_1_list[cc])
            teacher_incorrect_ratio_clipped_0_1_list = np.array([]) if np.all(cc) or not self.cfg.use_topr else np.array(teacher_ratio_clipped_0_1_list[ic])
            student_correct_ratio_clipped_0_1_list = np.array([]) if np.all(ic) or not self.cfg.use_topr else np.array(student_ratio_clipped_0_1_list[cc])
            student_incorrect_ratio_clipped_0_1_list = np.array([]) if np.all(cc) or not self.cfg.use_topr else np.array(student_ratio_clipped_0_1_list[ic])

            log_dict = {
                "avg_student_teacher_kl": avg_student_teacher_kl,
                "avg_student_teacher_kl_max": avg_student_teacher_kl_max,
                "avg_match_reward": avg_match_reward,
                "avg_correct_match_reward": 0 if len(correct_match_reward_trainer) == 0 else np.mean(correct_match_reward_trainer).item(),
                "avg_incorrect_match_reward": 0 if len(incorrect_match_reward_trainer) == 0 else np.mean(incorrect_match_reward_trainer).item(),
                "avg_ss_reward_mean": 0 if len(ss_reward_mean_list) == 0 else np.mean(ss_reward_mean_list).item(),
                "avg_ss_reward_min": 0 if len(ss_reward_min_list) == 0 else np.mean(ss_reward_min_list).item(),
                "avg_ss_reward": 0 if len(ss_reward_list) == 0 else np.mean(ss_reward_list).item(),
                "avg_correct_kl_mean": 0 if len(correct_kl_mean_list) == 0 else np.mean(correct_kl_mean_list).item(),
                "avg_incorrect_kl_mean": 0 if len(incorrect_kl_mean_list) == 0 else np.mean(incorrect_kl_mean_list).item(),
                "avg_correct_kl_sum": 0 if len(correct_kl_sum_list) == 0 else np.mean(correct_kl_sum_list).item(),
                "avg_incorrect_kl_sum": 0 if len(incorrect_kl_sum_list) == 0 else np.mean(incorrect_kl_sum_list).item(),
                "avg_correct_kl_max": 0 if len(correct_kl_max_list) == 0 else np.mean(correct_kl_max_list).item(),
                "avg_incorrect_kl_max": 0 if len(incorrect_kl_max_list) == 0 else np.mean(incorrect_kl_max_list).item(),
                "avg_correct_ss_reward_mean": 0 if len(correct_ss_reward_mean_list) == 0 else np.mean(correct_ss_reward_mean_list).item(),
                "avg_incorrect_ss_reward_mean": 0 if len(incorrect_ss_reward_mean_list) == 0 else np.mean(incorrect_ss_reward_mean_list).item(),
                "avg_correct_ss_reward_min": 0 if len(correct_ss_reward_min_list) == 0 else np.mean(correct_ss_reward_min_list).item(),
                "avg_incorrect_ss_reward_min": 0 if len(incorrect_ss_reward_min_list) == 0 else np.mean(incorrect_ss_reward_min_list).item(),
                "avg_incorrect_incorect": 0 if len(ii) == 0 else np.mean(ii).item(),
                "avg_teacher_correct_alpha": 0 if len(teacher_correct_ratio_clipped_0_1_list) == 0 else np.mean(teacher_correct_ratio_clipped_0_1_list).item(),
                "avg_teacher_incorrect_alpha": 0 if len(teacher_incorrect_ratio_clipped_0_1_list) == 0 else np.mean(teacher_incorrect_ratio_clipped_0_1_list).item(),
                "avg_student_correct_alpha": 0 if len(student_correct_ratio_clipped_0_1_list) == 0 else np.mean(student_correct_ratio_clipped_0_1_list).item(),
                "avg_student_incorrect_alpha": 0 if len(student_incorrect_ratio_clipped_0_1_list) == 0 else np.mean(student_incorrect_ratio_clipped_0_1_list).item(),
            }

            for k, v in log_dict.items():
                self.writer.add_scalar(k, v, self.global_step)

            logger.info(f"avg_student_teacher_kl: {avg_student_teacher_kl} avg_student_teacher_kl_max: {avg_student_teacher_kl_max} avg_match_reward {avg_match_reward}")

            async with Timer(f"computing GRPO normalized rewards"):
                if self.cfg.use_grpo and not self.cfg.remove_teacher_grpo_normalization:
                    teacher_prompt_idx = 0
                    teacher_score_sum = 0
                    for teacher_exp in teacher_experiences:
                        assert len(teacher_exp.info['custom_rewards']) == len(teacher_exp.num_actions[0]), "teacher_exp.info['custom_rewards'] must be equal to teacher_exp.num_actions[0]"
                        for i in range(len(teacher_exp.num_actions[0])):
                            prompt = all_teacher_prompts[teacher_prompt_idx]
                            teacher_score = final_reward_list[teacher_prompt_idx].item()

                            # logger.info(f"teacher_score {teacher_score}")
                            teacher_score -= np.mean(teacher_pass_at_n_dict[prompt])
                            # logger.info(f"np.mean(teacher_pass_at_n_dict[prompt]) {np.mean(teacher_pass_at_n_dict[prompt])} {len(teacher_pass_at_n_dict[prompt])}")
                            if teacher_std := np.std(teacher_pass_at_n_dict[prompt]) > 0:
                                teacher_score /= teacher_std

                            teacher_score_sum += teacher_score

                            # logger.info(f"2 teacher_score {teacher_score}")
                            # logger.info(f"teacher_exp.info['custom_rewards'][i][-1] {teacher_exp.info['custom_rewards'][i][-1]}")
                            teacher_exp.info['custom_rewards'][i][-1] = teacher_score

                            # logger.info(f"2 teacher_exp.info['custom_rewards'][i][-1] {teacher_exp.info['custom_rewards'][i][-1]}")
                            teacher_prompt_idx += 1

                    assert teacher_prompt_idx == len(all_teacher_prompts) == len(final_reward_list), "last teacher prompt idx must be equal to all teacher prompts length"

                    self.writer.add_scalar("avg_teacher_reward", teacher_score_sum / len(all_teacher_prompts), self.global_step)
                    logger.info(f"avg_teacher_reward: {teacher_score_sum / len(all_teacher_prompts)}")

        # 2 vis student and teacher to wandb and writer
        for experiences, prefix in zip([teacher_experiences, student_experiences], ["teacher", "student"]):
            vis = self._detokenize(experiences[0].sequences[0][: int(experiences[0].info["total_length"].flatten()[0])])
            self.writer.add_text(f"{prefix}_sequences", vis, self.global_step)

            if wandb.run is not None:
                data = []
                # Log up to 5 examples from different experiences
                for i in range(min(5, len(experiences))):
                    if len(experiences[i].sequences) > 0:
                        vis_example = self._detokenize(experiences[i].sequences[0][: int(experiences[i].info["total_length"].flatten()[0])])

                        # Extract student prompt and reasoning parts
                        if "Assistant: <think>" in vis_example:
                            prompt_part = vis_example.split("Assistant: <think>")[0]
                            reasoning_part = vis_example.split("Assistant: <think>")[1]
                        else:
                            prompt_part = vis_example
                            reasoning_part = ""

                        data.append([prompt_part, reasoning_part])

                if data:
                    wandb.log({
                        "step": self.global_step,
                        f"{prefix}_examples": wandb.Table(
                            columns=["prompt", "reasoning"],
                            data=data)
                    })

        self.writer.flush()

        # 3. calculate advantages and returns / along with tensorboard logging
        for experiences, buffer, prefix in zip([student_experiences, teacher_experiences],
                                              [self.student_replay_buffer, self.teacher_replay_buffer],
                                              ["student", "teacher"]):
            avg_rewards = 0
            avg_kl = 0
            avg_kl_max = 0
            avg_response_length = 0
            avg_orm_score = 0
            avg_custom_rewards = 0
            avg_advantages = 0
            avg_advantages_abs = 0

            async with Timer(f"Calculate {prefix} advantages and returns"):
                adv_tasks = []
                for experience in experiences:
                    adv_tasks.append(self._calc_advantages_and_returns(experience))

                for tsk in asyncio.as_completed(adv_tasks):
                    experience, metrics = await tsk
                    avg_rewards += metrics["avg_rewards"]
                    avg_kl += metrics["avg_kl"]
                    avg_kl_max += metrics["avg_kl_max"]
                    avg_response_length += metrics["avg_response_length"]
                    avg_orm_score += metrics["avg_orm_score"]
                    avg_custom_rewards += metrics["avg_custom_rewards"]
                    avg_advantages += metrics["avg_advantages"]
                    avg_advantages_abs += metrics["avg_advantages_abs"]
                    buffer.append(experience)

            # 4. tensorboard logging for this prefix
            if len(experiences) > 0:
                logger.info(f"{prefix.upper()} - avg_raw_rewards: {avg_rewards / len(experiences)}, avg_kl: {avg_kl / len(experiences)}, avg_response_length: {avg_response_length / len(experiences)}, avg_orm_score: {avg_orm_score / len(experiences)}, avg_custom_rewards: {avg_custom_rewards / len(experiences)}")
                self.writer.add_scalar(f"{prefix}_avg_raw_rewards", avg_rewards / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_kl", avg_kl / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_kl_max", avg_kl_max / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_response_length", avg_response_length / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_orm_score", avg_orm_score / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_custom_rewards", avg_custom_rewards / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_raw_advantages", avg_advantages / len(experiences), self.global_step)
                self.writer.add_scalar(f"{prefix}_avg_raw_advantages_abs", avg_advantages_abs / len(experiences), self.global_step)

        self.writer.flush()

    @torch.no_grad()
    async def inference_and_calculates(
            self,
            sequences_all: List[torch.Tensor],
            attention_mask_all: List[torch.Tensor],
            action_mask_all: Optional[List[torch.Tensor]],
            num_actions_all: Optional[List[int]],
            packed_seq_lens_all: Optional[List[int]],
            custom_rewards_all: Optional[List[torch.Tensor]],
            use_teacher_model: bool = False,
    ):
        num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
        num_critic_dp_groups = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
        num_ref_dp_groups = self.cfg.ref_num_nodes * self.cfg.ref_num_gpus_per_node
        num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node

        async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
            dp_iterator = self._split_dp_batch(
                (sequences, num_actions, attention_mask, packed_seq_lens),
                num_dps,
            )
            dp_tasks = []
            for dp_rank, (
                    micro_sequences,
                    micro_num_actions,
                    micro_attention_mask,
                    micro_packed_seq_lens,
            ) in enumerate(dp_iterator):
                model = self._get_dp_group_models(dp_rank, model_type)

                async def forward_fn(
                        local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                ):
                    return await local_model.forward.remote(
                        sequences=fwd_sequences,
                        num_actions=fwd_num_actions,
                        attention_mask=fwd_attention_mask,
                        packed_seq_lens=fwd_packed_seq_lens,
                    )

                dp_tasks.append(
                    self._split_and_run_micro_batch(
                        partial(forward_fn, model),
                        (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                        self.cfg.micro_forward_batch_size,
                    )
                )
            results = await asyncio.gather(*dp_tasks)
            results = sum(results, [])
            return results

        if action_mask_all is not None:
            num_actions_all = action_mask_all.size(1)

        # calculate critic values
        if self.cfg.colocate_all and self.critic_model is not None:
            await self.critic_model.backload_to_gpu()

        if self.critic_model is not None:
            value_ref = micro_infer_model(
                num_critic_dp_groups,
                "critic_model",
                sequences_all,
                num_actions_all,
                attention_mask_all,
                packed_seq_lens_all,
            )
            values = None
            if self.cfg.colocate_all:
                values = await value_ref
                await self.critic_model.offload_to_cpu()

        # calculate ref log probs
        base_action_log_probs_ref = micro_infer_model(
            num_ref_dp_groups, "ref_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
        )
        base_log_probs = None

        # handle colocate critic and reward model
        if self.cfg.colocate_critic_reward and not self.cfg.colocate_all and self.critic_model is not None:
            values = await value_ref
            await self.critic_model.async_run_method("empty_cache")

        # handle colocate actor and ref model
        if self.cfg.colocate_actor_ref or self.cfg.colocate_all:
            base_log_probs = await base_action_log_probs_ref
            await self.ref_model.async_run_method("empty_cache")

        # calculate rewards
        reward_refs = []
        logger.info(f"self.cfg.use_orm_score {self.cfg.use_orm_score} self.reward_model {self.reward_model}")
        if self.cfg.use_orm_score and self.reward_model:
            reward_refs.append(
                micro_infer_model(
                    num_reward_dp_groups,
                    "reward_model",
                    sequences_all,
                    num_actions_all,
                    attention_mask_all,
                    packed_seq_lens_all,
                )
            )

        if self.cfg.colocate_all:
            rewards = await asyncio.gather(*reward_refs)

        # calculate action log probs
        if self.cfg.colocate_all:
            if not use_teacher_model:
                await self.policy_model.backload_to_gpu()
            else:
                await self.teacher_model.backload_to_gpu()

        action_log_probs_ref = micro_infer_model(
            num_policy_dp_groups,
            "policy_model" if not use_teacher_model else "teacher_model",
            sequences_all,
            num_actions_all,
            attention_mask_all,
            packed_seq_lens_all,
        )
        action_log_probs = None
        if self.cfg.colocate_all:
            action_log_probs = await action_log_probs_ref
            if not use_teacher_model:
                await self.policy_model.offload_to_cpu()
            else:
                await self.teacher_model.offload_to_cpu()

        # wait all models done
        # if not colocate_actor_ref, then need to gather base_log_probs
        # if not colocate_critic_reward and self.critic_model is not None, then need to gather value
        # reward_refs is always handled at last
        if not self.cfg.colocate_all:
            if not self.cfg.colocate_actor_ref:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(
                        value_ref, base_action_log_probs_ref, action_log_probs_ref, *reward_refs
                    )
                    values, base_log_probs, action_log_probs, rewards = results[0], results[1], results[2], results[3:]
                else:
                    results = await asyncio.gather(base_action_log_probs_ref, action_log_probs_ref, *reward_refs)
                    base_log_probs, action_log_probs, rewards = results[0], results[1], results[2:]
            else:
                if not self.cfg.colocate_critic_reward and self.critic_model is not None:
                    results = await asyncio.gather(value_ref, action_log_probs_ref, *reward_refs)
                    values, action_log_probs, rewards = results[0], results[1], results[2:]
                else:
                    results = await asyncio.gather(action_log_probs_ref, *reward_refs)
                    action_log_probs, rewards = results[0], results[1:]

        r = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else None
        if not self.cfg.colocate_all:
            empty_cache_tasks = [
                self.policy_model.async_run_method("empty_cache") if not use_teacher_model else self.teacher_model.async_run_method("empty_cache"),
                self.ref_model.async_run_method("empty_cache"),
            ]
            if self.critic_model:
                empty_cache_tasks.append(self.critic_model.async_run_method("empty_cache"))
            if self.reward_model:
                empty_cache_tasks.extend([rm.async_run_method("empty_cache") for rm in self.reward_model])
            await asyncio.gather(*empty_cache_tasks)

        # 6. calculate kl divergence
        experiences = []
        if self.critic_model is not None:
            values = values[: len(sequences_all)]
        base_log_probs = base_log_probs[: len(sequences_all)]
        action_log_probs = action_log_probs[: len(sequences_all)]
        if r is not None:
            r = r[: len(sequences_all)]
        for i in range(len(action_log_probs)):
            response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
            total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
            kl = compute_approx_kl(
                action_log_probs[i],
                base_log_probs[i],
                action_mask=None,
                use_kl_estimator_k3=self.cfg.use_kl_estimator_k3,
                use_abs_kl=self.cfg.use_abs_kl,
            )
            kl_max = torch.max(kl.abs(), dim=-1)[0]
            kl_mean = masked_mean(kl, None, dim=-1)
            if r is not None:
                local_reward = r[i]
            else:
                local_reward = None
            info = {
                "kl": kl_mean,
                "kl_max": kl_max,
                "reward": local_reward,
                "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
                "response_length": response_length,
                "total_length": total_length,
                "num_actions": num_actions_all[i],
            }
            experiences.append(
                Experience(
                    sequences_all[i],
                    action_log_probs[i],
                    base_log_probs[i],
                    values[i] if self.critic_model is not None else None,
                    None,
                    None,
                    attention_mask_all[i],
                    None,
                    response_length,
                    torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),
                    info,
                    kl,
                    torch.ones_like(action_log_probs[i]) if self.cfg.use_topr else None,
                )
            )
        return experiences

    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        responses, _ = await gen_func(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        return responses

    def build_dataloader(self, dataset):
        # prepare dataloader
        prompts_dataloader = DataLoader(
            dataset, batch_size=self.cfg.rollout_batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8
        )
        self.num_update_steps_per_episodes = (
            len(dataset) * self.cfg.n_samples_per_prompt // self.cfg.train_batch_size * self.cfg.max_epochs
        )
        max_steps = math.ceil(self.cfg.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps
        return prompts_dataloader

    async def build_models(self, PolicyRayActor, CriticRayActor, RefRayActor, RewardRayActor=None):
        cfg = self.cfg
        pg = None

        if cfg.colocate_all:
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            
            # Create separate teacher model if flag is enabled
            if cfg.separate_teacher_model:
                teacher_model = PPORayActorGroup(
                    cfg.actor_num_nodes,
                    cfg.actor_num_gpus_per_node,
                    PolicyRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                )
            else:
                teacher_model = policy_model  # Share the same model
            
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.2,
                        )
                    )
            else:
                reward_models = None

        else:
            if cfg.colocate_actor_ref:
                assert (
                    cfg.actor_num_nodes == cfg.ref_num_nodes
                    and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [
                    {"GPU": cfg.actor_num_gpus_per_node, "CPU": cfg.actor_num_gpus_per_node}
                    for _ in range(cfg.actor_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )
            
            # Create separate teacher model if flag is enabled
            if cfg.separate_teacher_model:
                teacher_model = PPORayActorGroup(
                    cfg.actor_num_nodes,
                    cfg.actor_num_gpus_per_node,
                    PolicyRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                )
            else:
                teacher_model = policy_model  # Share the same model
            
            ref_model = PPORayActorGroup(
                cfg.ref_num_nodes,
                cfg.ref_num_gpus_per_node,
                RefRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )

            # if colocated, create placement group for critic and reward model explicitly.
            pg = None
            if cfg.colocate_critic_reward:
                assert (
                    cfg.critic_num_nodes == cfg.reward_num_nodes
                    and cfg.critic_num_gpus_per_node == cfg.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

                bundles = [
                    {"GPU": cfg.critic_num_gpus_per_node, "CPU": cfg.critic_num_gpus_per_node}
                    for _ in range(cfg.critic_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            if cfg.critic_pretrain:
                critic_model = PPORayActorGroup(
                    cfg.critic_num_nodes,
                    cfg.critic_num_gpus_per_node,
                    CriticRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                )
            else:
                critic_model = None

            # multiple reward models
            if RewardRayActor is not None and cfg.reward_pretrain:
                reward_pretrains = cfg.reward_pretrain.split(",")
                reward_models = []
                for _ in reward_pretrains:
                    reward_models.append(
                        PPORayActorGroup(
                            cfg.reward_num_nodes,
                            cfg.reward_num_gpus_per_node,
                            RewardRayActor,
                            pg=pg,
                            num_gpus_per_actor=0.25 if pg else 1,
                        )
                    )
            else:
                reward_models = None

        if not cfg.colocate_all:
            refs = []
            refs.extend(ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            if cfg.separate_teacher_model:
                refs.extend(teacher_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            if cfg.critic_pretrain:
                refs.extend(critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    refs.extend(reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))
            await asyncio.gather(*refs)
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
            if cfg.separate_teacher_model:
                await teacher_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
        else:
            await asyncio.gather(*ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
            await policy_model.offload_to_cpu()
            if cfg.separate_teacher_model:
                await asyncio.gather(*teacher_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
                await teacher_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
                await teacher_model.offload_to_cpu()
            if cfg.critic_pretrain:
                await asyncio.gather(*critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
                await critic_model.offload_to_cpu()
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    await asyncio.gather(*reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))

        self.policy_model = policy_model
        self.teacher_model = teacher_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/teacher/ref/critic/reward models done")

    async def ppo_local_train_policy(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int, prefix: str = "", backlog: bool = False):
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer(f"{prefix.capitalize()} Policy model training"):
                model = self.teacher_model if "teacher" in prefix.lower() else self.policy_model
                status = await model.async_ppo_train(global_steps, replay_buffers)

            # Log with prefix for separate tracking
            metric_prefix = f"{prefix}_" if prefix else ""
            self.writer.add_scalar(f"{metric_prefix}ppo_clip_count", status[0]["clip_ratio"], global_steps)
            self.writer.add_scalar(f"{metric_prefix}policy_update_steps", status[0]["policy_update_steps"], global_steps)
            self.writer.add_scalar(f"{metric_prefix}policy_entropy", status[0]["entropy"], global_steps)
            await model.async_run_method("empty_cache")

        if global_steps > self.cfg.freezing_actor_steps:
            return status[0]

    async def ppo_local_train_critic(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int, prefix: str = ""):
        async with Timer(f"{prefix.capitalize()} Critic model training"):
            status = await self.critic_model.async_ppo_train(global_steps, replay_buffers)
        if critic_loss := status[0].get("critic_loss", None):
            # Log with prefix for separate tracking
            metric_prefix = f"{prefix}_" if prefix else ""
            self.writer.add_scalar(f"{metric_prefix}critic_loss", critic_loss, global_steps)
            self.writer.add_scalar(f"{metric_prefix}critic_update_steps", status[0]["critic_update_steps"], global_steps)
        return status[0]

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        raise NotImplementedError("custom reward function is not supported yet")

    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            experience.action_mask,
            num_actions,
            self.cfg.gamma,
            self.cfg.lambd,
            packing=True,
        )

        return_sums = reward.sum(dim=-1)
        return_sums /= len(num_actions)
        experience.info["return"] = return_sums
        experience.kl = None

        avg_rewards = return_sums.mean().item()
        avg_kl = experience.info["kl"].mean().item()
        avg_kl_max = experience.info["kl_max"].mean().item()

        avg_response_length = experience.info["response_length"].mean().item()
        if experience.info["reward"] is not None:
            avg_orm_score = experience.info["reward"].mean().item()
        else:
            avg_orm_score = 0

        if experience.info["custom_rewards"] is not None:

            def func(x):
                return [r.sum() for r in x]

            avg_custom_rewards = torch.stack(func(experience.info["custom_rewards"])).mean().item()
            # experience.info["avg_custom_rewards"] = torch.stack(func(experience.info["custom_rewards"]))
        else:
            avg_custom_rewards = 0

        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        del experience.info["kl_max"]
        experience.to_device("cpu")

        # for replay buffer split batch
        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        metrics = {
            "avg_rewards": avg_rewards,
            "avg_kl": avg_kl,
            "avg_kl_max": avg_kl_max,
            "avg_response_length": avg_response_length,
            "avg_orm_score": avg_orm_score,
            "avg_custom_rewards": avg_custom_rewards,
            "avg_advantages": experience.advantages.mean().item(),
            "avg_advantages_abs": experience.advantages.abs().mean().item(),
        }

        return experience, metrics

    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        # This function is used when not packing samples
        # concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        prompt_token_lens, response_token_lens = [], []
        inputs_token_ids, outputs_token_ids = [], []
        for prompt, output in zip(prompts, outputs):
            input_token_ids = self._tokenize(prompt, self.cfg.prompt_max_len, padding=False)["input_ids"]
            response_token_ids = self._tokenize(output, self.cfg.generate_max_len, padding=False)["input_ids"]

            inputs_token_ids.append(input_token_ids)
            outputs_token_ids.append(response_token_ids)

            prompt_token_len = len(input_token_ids)
            response_token_len = len(response_token_ids)
            prompt_token_lens.append(prompt_token_len)
            response_token_lens.append(response_token_len)

            max_input_len = max(max_input_len, prompt_token_len)
            max_output_len = max(max_output_len, response_token_len)

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for i, prompt in enumerate(prompts):
            # left padding input
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # right padding output
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # replace last token with eos_token_id if it is not eos_token_id, keep the total length of output_ids
            # output_ids[output_len - 1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences, attention_mask, action_mask

    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str],
        teacher_prompts: List[str],
        outputs: List[str],
        custom_rewards: Optional[List[torch.Tensor]],
        teacher_custom_rewards: Optional[List[torch.Tensor]],
        packing_max_len: int,

    ):
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        if teacher_custom_rewards is not None:
            ret_teacher_custom_rewards = []
        else:
            ret_teacher_custom_rewards = None
        
        # Teacher sequences (always provided)
        ret_teacher_sequences = []
        ret_teacher_attention_masks = []
        ret_teacher_num_actions = []
        ret_teacher_packed_seq_lens = []

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0 and len(teacher_prompts) == len(prompts)
        ), "prompts, outputs, and teacher_prompts must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            teacher_rewards = [] if teacher_custom_rewards else None
            seq_offset = 0
            seq_index = 0
            
            # Teacher sequence variables
            out_teacher_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_teacher_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_teacher_num_actions = []
            out_teacher_packed_seq_lens = []
            teacher_seq_offset = 0
            
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                teacher_rewards,
                seq_offset,
                seq_index,
                out_teacher_sequence,
                out_teacher_attention_mask,
                out_teacher_num_actions,
                out_teacher_packed_seq_lens,
                teacher_seq_offset,
            )

        def _accumulate(
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
            sequence,
            attention_mask,
            num_action,
            total_len,
            custom_rewards,
            i,
            # Teacher sequence parameters
            out_teacher_sequence,
            out_teacher_attention_mask,
            out_teacher_num_actions,
            out_teacher_packed_seq_lens,
            teacher_rewards,
            teacher_seq_offset,
            teacher_sequence,
            teacher_attention_mask,
            teacher_num_action,
            teacher_total_len,
            teacher_custom_rewards,
        ):
            # Student sequence
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            
            # Teacher sequence
            out_teacher_sequence[teacher_seq_offset : teacher_seq_offset + teacher_total_len] = torch.tensor(teacher_sequence)
            out_teacher_attention_mask[teacher_seq_offset : teacher_seq_offset + teacher_total_len] = seq_index + 1
            out_teacher_num_actions.append(teacher_num_action)
            out_teacher_packed_seq_lens.append(teacher_total_len)
            if teacher_custom_rewards:
                teacher_rewards.append(teacher_custom_rewards[i])
            
            return seq_offset + total_len, seq_index + 1, teacher_seq_offset + teacher_total_len

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []
        
        # Teacher sequences
        teacher_sequences = []
        teacher_attention_masks = []
        teacher_num_actions = []
        teacher_total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]
        teacher_input_token_ids = self._tokenize(teacher_prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]

        for input_ids, response_ids, teacher_input_ids in zip(input_token_ids, response_token_ids, teacher_input_token_ids):
            # Student sequences
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))
            
            # Teacher sequences (teacher prompt + same response)
            teacher_sequences.append(teacher_input_ids + response_ids)
            teacher_attention_masks.append(torch.ones((len(teacher_input_ids) + len(response_ids),), dtype=torch.float32))
            teacher_num_actions.append(len(response_ids))
            teacher_total_lens.append(len(teacher_input_ids) + len(response_ids))

        # make packed sequences
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            teacher_rewards,
            seq_offset,
            seq_index,
            out_teacher_sequence,
            out_teacher_attention_mask,
            out_teacher_num_actions,
            out_teacher_packed_seq_lens,
            teacher_seq_offset,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len, teacher_sequence, teacher_attention_mask, teacher_num_action, teacher_total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens, teacher_sequences, teacher_attention_masks, teacher_num_actions, teacher_total_lens)
        ):
            if seq_offset + total_len < packing_max_len and teacher_seq_offset + teacher_total_len < packing_max_len:
                seq_offset, seq_index, teacher_seq_offset = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                    out_teacher_sequence,
                    out_teacher_attention_mask,
                    out_teacher_num_actions,
                    out_teacher_packed_seq_lens,
                    teacher_rewards,
                    teacher_seq_offset,
                    teacher_sequence,
                    teacher_attention_mask,
                    teacher_num_action,
                    teacher_total_len,
                    teacher_custom_rewards,
                )
            elif max(seq_offset + total_len, teacher_seq_offset + teacher_total_len) == packing_max_len:
                seq_offset, seq_index, teacher_seq_offset = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                    out_teacher_sequence,
                    out_teacher_attention_mask,
                    out_teacher_num_actions,
                    out_teacher_packed_seq_lens,
                    teacher_rewards,
                    teacher_seq_offset,
                    teacher_sequence,
                    teacher_attention_mask,
                    teacher_num_action,
                    teacher_total_len,
                    teacher_custom_rewards,
                )
                # Pack student sequences
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                
                # Pack teacher sequences
                valid_teacher_size = out_teacher_attention_mask.nonzero().size(0)
                ret_teacher_sequences.append(out_teacher_sequence[:valid_teacher_size].unsqueeze(0))
                ret_teacher_attention_masks.append(out_teacher_attention_mask[:valid_teacher_size].unsqueeze(0))
                ret_teacher_num_actions.append(out_teacher_num_actions)
                ret_teacher_packed_seq_lens.append(out_teacher_packed_seq_lens)
                if teacher_custom_rewards:
                    ret_teacher_custom_rewards.append(teacher_rewards)
                
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    teacher_rewards,
                    seq_offset,
                    seq_index,
                    out_teacher_sequence,
                    out_teacher_attention_mask,
                    out_teacher_num_actions,
                    out_teacher_packed_seq_lens,
                    teacher_seq_offset,
                ) = _new_instance()
            elif max(seq_offset + total_len, teacher_seq_offset + teacher_total_len) > packing_max_len:
                if seq_offset > 0:
                    # Pack student sequences
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    
                    # Pack teacher sequences
                    valid_teacher_size = out_teacher_attention_mask.nonzero().size(0)
                    ret_teacher_sequences.append(out_teacher_sequence[:valid_teacher_size].unsqueeze(0))
                    ret_teacher_attention_masks.append(out_teacher_attention_mask[:valid_teacher_size].unsqueeze(0))
                    ret_teacher_num_actions.append(out_teacher_num_actions)
                    ret_teacher_packed_seq_lens.append(out_teacher_packed_seq_lens)
                    if teacher_custom_rewards:
                        ret_teacher_custom_rewards.append(teacher_rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        teacher_rewards,
                        seq_offset,
                        seq_index,
                        out_teacher_sequence,
                        out_teacher_attention_mask,
                        out_teacher_num_actions,
                        out_teacher_packed_seq_lens,
                        teacher_seq_offset,
                    ) = _new_instance()
                    seq_offset, seq_index, teacher_seq_offset = _accumulate(
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                        sequence,
                        attention_mask,
                        num_action,
                        total_len,
                        custom_rewards,
                        i,
                        out_teacher_sequence,
                        out_teacher_attention_mask,
                        out_teacher_num_actions,
                        out_teacher_packed_seq_lens,
                        teacher_rewards,
                        teacher_seq_offset,
                        teacher_sequence,
                        teacher_attention_mask,
                        teacher_num_action,
                        teacher_total_len,
                        teacher_custom_rewards
                    )

        if seq_offset > 0:
            # Pack final student sequences
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)
            
            # Pack final teacher sequences
            valid_teacher_size = out_teacher_attention_mask.nonzero().size(0)
            ret_teacher_sequences.append(out_teacher_sequence[:valid_teacher_size].unsqueeze(0))
            ret_teacher_attention_masks.append(out_teacher_attention_mask[:valid_teacher_size].unsqueeze(0))
            ret_teacher_num_actions.append(out_teacher_num_actions)
            ret_teacher_packed_seq_lens.append(out_teacher_packed_seq_lens)
            if teacher_custom_rewards:
                ret_teacher_custom_rewards.append(teacher_rewards)

            assert (len(ret_custom_rewards) == len(ret_teacher_custom_rewards)), "Number of packed student and teacher rewards must be the same"

        return (ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards, 
                ret_teacher_sequences, ret_teacher_attention_masks, ret_teacher_num_actions, ret_teacher_packed_seq_lens, ret_teacher_custom_rewards)

    def _get_dp_group_models(self, dp_rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[dp_rank]

    def _split_dp_batch(self, batch, num_dp, drop_last=False):
        # Convert batch tuple to list of lists, handling None values
        batch_lists = []
        batch_size = None
        for item in batch:
            if item is not None:
                if batch_size is None:
                    batch_size = len(item)
                batch_lists.append(item)
            else:
                batch_lists.append(None)

        if drop_last:
            dp_size = batch_size // num_dp
        else:
            dp_size = (batch_size + num_dp - 1) // num_dp
        valid_size = dp_size * num_dp

        if not drop_last:
            padding_index = None
            for i in range(len(batch_lists)):
                if batch_lists[i] is not None and (
                    isinstance(batch_lists[i], torch.Tensor) or isinstance(batch_lists[i], list)
                ):
                    padding_size = valid_size - len(batch_lists[i])
                    if padding_size > 0:
                        if padding_index is None:
                            if padding_size > len(batch_lists[i]):
                                padding_index = random.choices(range(len(batch_lists[i])), k=padding_size)
                            else:
                                padding_index = random.sample(range(len(batch_lists[i])), padding_size)
                        if isinstance(batch_lists[i], torch.Tensor):
                            batch_lists[i] = torch.cat([batch_lists[i], batch_lists[i][padding_index]], dim=0)
                        elif isinstance(batch_lists[i], list):
                            batch_lists[i] = batch_lists[i] + [batch_lists[i][j] for j in padding_index]

        for i in range(num_dp):
            # Extract micro batch for each input list
            micro_batch = []
            for batch_list in batch_lists:
                if batch_list is None:
                    micro_batch.append(None)
                elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                    micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
                else:
                    micro_batch.append(batch_list)
            yield tuple(micro_batch)

    def _split_dp_batch_dynamic_balance(self, batch, num_dp, balanced_values):
        batch = list(batch)
        assert len(batch) == len(balanced_values), "batch and balanced_values must have the same length"
        results = self._split_weighted_objects(zip(balanced_values, batch), num_dp)
        # re organize to the original format
        for i in range(num_dp):
            ret = [[] for _ in range(len(results[i][0]))]
            for sample in results[i]:
                for j, v in enumerate(sample):
                    ret[j].append(v)
            yield ret

    def _split_weighted_objects(self, items, n):
        result = [[] for _ in range(n)]

        heap = [(0, i) for i in range(n)]
        heapify(heap)

        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        for weight, obj in sorted_items:
            current_sum, index = heappop(heap)
            result[index].append(obj)
            heappush(heap, (current_sum + weight, index))

        return result

    async def _split_and_run_micro_batch(self, async_fn, batch_args, micro_size):
        # Ensure batch_args is a sequence of lists with equal length
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches
        for i in range(0, batch_size, micro_size):
            # Take slice i:i+micro_size from each argument
            micro_batch_args = []
            for arg in batch_args:
                if arg is not None:
                    if not isinstance(arg, torch.Tensor) and not isinstance(arg, list):
                        micro_batch_args.append(arg)
                    elif micro_size > 1 or isinstance(arg, torch.Tensor):
                        micro_batch_args.append(arg[i : i + micro_size])
                    else:
                        micro_batch_args.append(arg[i])
                else:
                    micro_batch_args.append(None)
            results.append(await async_fn(*micro_batch_args))
        return results

    def _get_generate_function(self, dp_rank: int):
        llm = self.vllm_engines[dp_rank % len(self.vllm_engines)]

        async def generate(prompts: List[str], truncate_prompt=True, **kwargs):
            if truncate_prompt:
                prompt_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
            else:
                prompt_token_ids = self._tokenize(prompts, padding=False)["input_ids"]
            outputs = await llm.generate.remote(prompt_token_ids=prompt_token_ids, **kwargs)
            responses = []
            prompt_logprobs = []
            finish_reasons = []
            for i, prompt in enumerate(prompts):
                content = outputs[i].outputs[0].text
                finish_reasons.append(outputs[i].outputs[0].finish_reason)
                responses.append(content)
                if outputs[i].prompt_logprobs:
                    prompt_logprobs.append(outputs[i].prompt_logprobs)
            if len(prompt_logprobs) > 0:
                return (
                    responses,
                    finish_reasons,
                    prompt_logprobs,
                )
            else:
                return responses, finish_reasons

        return generate

    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _warp_custom_reward_model_fn(self):
        if self.reward_model:
            # TODO: support multiple reward models]
            num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node

            async def warpped_reward_model_fn(prompts: List[str], outputs: List[str]):
                (
                    sequences,
                    attention_mask,
                    _,
                    packed_seq_lens,
                    _,
                ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                    prompts, outputs, None, self.cfg.packing_max_len
                )
                split_iterator = self._split_dp_batch(
                    (sequences, attention_mask, packed_seq_lens), num_policy_dp_groups
                )
                dp_tasks = []

                async def _rm_run(rm, seq, mask, lens):
                    return await rm.forward.remote(seq, mask, packed_seq_lens=lens)

                for dp_rank, args in enumerate(split_iterator):
                    rm = self._get_dp_group_models(dp_rank, "reward_model")
                    dp_tasks.append(
                        self._split_and_run_micro_batch(
                            partial(_rm_run, rm),
                            args,
                            self.cfg.micro_forward_batch_size,
                        )
                    )
                outputs = await asyncio.gather(*dp_tasks)
                outputs = sum(outputs, [])  # gather dp
                outputs = outputs[: len(sequences)]  # drop padding
                outputs = torch.hstack(outputs)

                assert outputs.size(0) == len(prompts), "reward outputs number must be equal to prompts number"
                return outputs

            return warpped_reward_model_fn
        else:
            return None

    async def _offload_vllm_engines(self):
        offload_tasks = []
        for engine in self.vllm_engines:
            offload_tasks.append(engine.offload_to_cpu.remote())
        await asyncio.gather(*offload_tasks)

    async def _backload_vllm_engines(self):
        backload_tasks = []
        for engine in self.vllm_engines:
            backload_tasks.append(engine.backload_to_gpu.remote())
        await asyncio.gather(*backload_tasks)

    async def _sync_policy_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.policy_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)
        # await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)


    async def _sync_teacher_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.teacher_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.teacher_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)