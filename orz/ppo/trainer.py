import asyncio
import json
import os
import random
from functools import partial
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict, Counter
import gc

import torch
from loguru import logger
from tqdm import tqdm

from orz.ppo.utils import (
    Timer,
    normalize_advantages,
)

from orz.ppo.base_trainer import BaseTrainer


class RayPPOTrainer(BaseTrainer):
    async def train(self):
        # 1. create rank0 policy model and vllm_engines groups, then boardcast weights to vllm engins
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()
            await self._backload_vllm_engines()

        async with Timer("Policy init vllm engines actor group"):
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
        self.student_training_step = 0
        self.teacher_training_step = 0
        # Cumulative counters across the whole run (never reset)
        self.student_steps_total = 0
        self.teacher_steps_total = 0
        # Warmup counter for initial teacher-only training rounds. This is used when cfg.initial_teacher_training_rounds > 0 to force a number of teacher updates before the regular teacher/student alternation.
        self.initial_teacher_training_step = 0
        sync_teacher_weigts = False
        start_episode = consumed_samples // self.cfg.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * self.cfg.rollout_batch_size)
        for episode in range(start_episode, self.cfg.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()), desc=f"Episode [{episode + 1}/{self.cfg.num_episodes}]"
            )
            for iter, rand_prompts in enumerate(self.prompts_dataloader):

                if self.cfg.eval_student:
                    if self.student_steps_total % self.cfg.eval_interval == 0:
                        async with Timer(f"Eval of the student model on global step {self.global_step}"):
                            # Ensure vLLM engines are set for student before syncing
                            await self._ensure_vllm_role("student")
                            await self._major_sync_policy_weights_to_vllm()
                            await self.eval(prefix="")
                            self.student_steps_total += 1 # ToDo: hack to avoid multiple evals per student training step

                if self.cfg.separate_teacher_model and self.cfg.eval_teacher:
                    if self.teacher_steps_total % self.cfg.eval_interval == 0:
                        async with Timer(f"Eval of the teacher model on global step {self.global_step}"):
                            # Switch vLLM engines to teacher if needed
                            await self._ensure_vllm_role("teacher")
                            await self._major_sync_teacher_weights_to_vllm()
                            await self.eval(prefix="teacher")
                            self.teacher_steps_total += 1 # ToDo: hack to avoid multiple evals per teacher training step

                # 2. determine what model to train
                self.train_teacher = False
                self.train_student = False

                # Optional teacher warmup: run N initial teacher-only rounds
                initial_teacher_training_rounds = self.cfg.initial_teacher_training_rounds
                if initial_teacher_training_rounds > 0 and self.initial_teacher_training_step < initial_teacher_training_rounds:
                    logger.info(
                        f"initial teacher warmup, {self.global_step} global step, {self.initial_teacher_training_step}/{initial_teacher_training_rounds}"
                    )
                    self.train_teacher = True
                    self.initial_teacher_training_step += 1
                    self.teacher_steps_total += 1
                else:
                    if self.cfg.student_training_rounds > 0 or self.cfg.teacher_training_rounds > 0:
                        if self.teacher_training_step < self.cfg.teacher_training_rounds:
                            logger.info(f'training teacher model, {self.global_step} global step, {self.teacher_training_step} teacher step')
                            self.train_teacher = True
                            self.teacher_training_step += 1
                            self.teacher_steps_total += 1
                        elif self.student_training_step < self.cfg.student_training_rounds:
                            logger.info(f'training student model, {self.global_step} global step, {self.student_training_step} student step')
                            self.train_student = True
                            self.student_training_step += 1
                            self.student_steps_total += 1
                            if self.student_training_step == self.cfg.student_training_rounds:
                                self.student_training_step = 0
                                self.teacher_training_step = 0
                    else:
                        # Train both teacher and student in the same global step
                        self.train_teacher = True
                        self.train_student = True
                        self.teacher_steps_total += 1
                        self.student_steps_total += 1

                logger.info(f'train_teacher {self.train_teacher}, train_student {self.train_student}')

                # 3. make experiences, calculate advantages and returns
                await self.make_experience(rand_prompts)

                # check if has enough data
                if len(self.student_replay_buffer) <= 0 or len(self.teacher_replay_buffer) <= 0:
                    await self._major_sync_policy_weights_to_vllm()
                    continue

                if self.cfg.advantage_normalize:
                    self.student_replay_buffer = normalize_advantages(self.student_replay_buffer)
                    self.teacher_replay_buffer = normalize_advantages(self.teacher_replay_buffer)

                if self.cfg.separate_teacher_model:
                    sfp = await self.policy_model.async_run_method("_weight_fingerprint")
                    tfp = await self.teacher_model.async_run_method("_weight_fingerprint")

                if self.train_teacher and not self.train_student:
                        train_set = zip([self.teacher_replay_buffer], ["teacher"])
                        self.student_replay_buffer.clear()
                elif not self.train_teacher and  self.train_student:
                    train_set = zip([self.student_replay_buffer], [""])
                    self.teacher_replay_buffer.clear()
                elif self.train_teacher and self.train_student:
                    if self.cfg.student_teacher_order:
                        train_set = zip([self.student_replay_buffer, self.teacher_replay_buffer], ["", 'teacher'])
                    else:
                        train_set = zip([self.teacher_replay_buffer, self.student_replay_buffer], ['teacher', ''])
                else:
                    raise ValueError("Either student or teacher must be trained in each iteration")

                if self.train_student and (self.cfg.skip_student_training_to_pretrain_teacher or self.global_step < self.cfg.skip_student_first_n_rounds):
                    logger.info("Skipping student training because of skip_student_training_to_pretrain_teacher flag")
                    train_set = zip([], [])
                    self.student_replay_buffer.clear()

                if self.cfg.critic_pretrain and self.cfg.colocate_critic_policy and self.cfg.offload_critic_policy_colocation:
                    await self.critic_model.offload_to_cpu()
                    await self.critic_model.backload_to_gpu()

                for replay_buffer, prefix in train_set:
                    logger.info(f"Start training {prefix} model, replay buffer size: {len(replay_buffer)}")
                    model = self.teacher_model if self.cfg.separate_teacher_model and prefix == "teacher" else self.policy_model
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
                            async with Timer(f"Critic {prefix} model training"):
                                await self.critic_model.backload_to_gpu()
                                await self.ppo_local_train_critic(critic_buffers, self.global_step, prefix)
                                await self.critic_model.offload_to_cpu()
                        async with Timer(f"Actor {prefix} model training"):
                            await model.backload_to_gpu()
                            status = await self.ppo_local_train_policy(model, policy_buffers, self.global_step, prefix)
                            await model.offload_to_cpu()
                    elif self.cfg.critic_pretrain and self.cfg.colocate_critic_policy and self.cfg.offload_critic_policy_colocation:
                        if self.critic_model is not None:
                            async with Timer(f"Critic {prefix} model training"):
                                await model.offload_to_cpu()
                                await self.ppo_local_train_critic(critic_buffers, self.global_step, prefix)
                                await model.backload_to_gpu()
                        async with Timer(f"Actor {prefix} model training"):
                            await self.critic_model.offload_to_cpu()
                            status = await self.ppo_local_train_policy(model, policy_buffers, self.global_step, prefix)
                            await self.critic_model.backload_to_gpu()
                    else:
                        if self.critic_model is not None:
                            async with Timer(f"Actor and Critic {prefix} model training"):
                                status = await asyncio.gather(
                                    self.ppo_local_train_policy(model, policy_buffers, self.global_step, prefix),
                                    self.ppo_local_train_critic(critic_buffers, self.global_step, prefix),
                                )
                                await asyncio.gather(
                                    model.async_run_method("empty_cache"),
                                    self.critic_model.async_run_method("empty_cache"),
                                )
                                status = status[0]
                        else:
                            async with Timer(f"Actor {prefix} model training"):
                                status = await self.ppo_local_train_policy(model, policy_buffers, self.global_step, prefix)
                                await model.async_run_method("empty_cache")

                    replay_buffer.clear()
                    gc.collect()

                    # 5. set logs
                    logger.info(f'Status {prefix} {status}')

                if not self.cfg.colocate_all:
                    await self.policy_model.offload_to_cpu()
                    await self.policy_model.async_run_method("empty_cache")
                    await self.policy_model.backload_to_gpu()

                    # if train_student and self.cfg.separate_teacher_model:
                    if self.cfg.separate_teacher_model:
                        await self.teacher_model.offload_to_cpu()
                        await self.teacher_model.async_run_method("empty_cache")
                        await self.teacher_model.backload_to_gpu()

                    if self.cfg.critic_pretrain:
                        await self.critic_model.offload_to_cpu()
                        await self.critic_model.async_run_method("empty_cache")
                        await self.critic_model.backload_to_gpu()

                if self.cfg.separate_teacher_model:
                    logger.info(f"Global step {self.global_step}, student_training_step {self.student_training_step}, teacher_training_step {self.teacher_training_step}, sync teacher weigts {sync_teacher_weigts}")
                    sfp2 = await self.policy_model.async_run_method("_weight_fingerprint")
                    tfp2 = await self.teacher_model.async_run_method("_weight_fingerprint")
                    logger.info(f"policy model weights changed: {sfp[0]['digest'] != sfp2[0]['digest']}")
                    logger.info(f"teacher model weights changed: {tfp[0]['digest'] != tfp2[0]['digest']}")
                    logger.info(f"policy and teacher model weights before training equal: {sfp[0]['digest'] == tfp[0]['digest']}")
                    logger.info(f"policy and teacher model weights after training equal: {sfp2[0]['digest'] == tfp2[0]['digest']}")

                pbar.update()
                # log epoch info
                self.writer.add_scalar("episode_idx", episode, self.global_step)
                self.writer.add_scalar("teacher_training", self.train_teacher, self.global_step)
                self.writer.add_scalar("student_training", self.train_student, self.global_step)
                self.writer.add_scalar("teacher_training_step", self.teacher_training_step, self.global_step)
                self.writer.add_scalar("student_training_step", self.student_training_step, self.global_step)
                self.writer.add_scalar("teacher_steps_total", self.teacher_steps_total, self.global_step)
                self.writer.add_scalar("student_steps_total", self.student_steps_total, self.global_step)
                self.global_step += 1
                if self.global_step % self.cfg.save_interval == 0:
                    await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                    if self.critic_model is not None:
                        await self.critic_model.async_save_model(self.tokenizer, self.global_step)
                    if self.cfg.separate_teacher_model:
                        await self.teacher_model.async_save_model(self.tokenizer, f'teacher-{self.global_step}')
                    logger.info("Successfully save model weights, training continue.")

                if self.cfg.separate_teacher_model and self.cfg.sync_teacher_weights and (self.global_step % self.cfg.synce_teacher_weights_interval) == 0: #(self.student_training_step == self.cfg.student_training_rounds):
                    async with Timer("Sync policy weights into teacher weights"):
                        await self._sync_policy_weights_to_teacher()
                        logger.info(f"Successfully loaded policy params to teacher, {self.global_step} global step")
                        # await self.teacher_model.async_run_method("empty_cache")
                    sync_teacher_weigts = True
                    logger.info("syncing teacher weigts checking")
                    checkpoint_path = os.path.join(self.cfg.save_path, "iter_current", "policy", 'model.safetensors')
                    chfp = await self.policy_model.async_run_method("_weight_fingerprint", checkpoint_path)
                    sfp3 = await self.policy_model.async_run_method("_weight_fingerprint")
                    tfp3 = await self.teacher_model.async_run_method("_weight_fingerprint")
                    logger.info(f"policy and teacher model weights are the same after syncing: {sfp3[0]['digest'] == tfp3[0]['digest']}")
                    logger.info(f"teacher and previous teacher model weights are the same after syncing: {tfp2[0]['digest'] == tfp3[0]['digest']}")
                    logger.info(f"policy and previous policy model weights are the same after syncing: {sfp2[0]['digest'] == sfp3[0]['digest']}")
                    logger.info(f"checkpoint and current teacher model: {chfp[0]['digest'] == tfp3[0]['digest']}")
                    logger.info(f"checkpoint and previous teacher model: {chfp[0]['digest'] == tfp2[0]['digest']}")
                    logger.info(f"checkpoint and previous policy model: {chfp[0]['digest'] == sfp2[0]['digest']}")
                    logger.info(f"checkpoint and current policy model: {chfp[0]['digest'] == sfp3[0]['digest']}")
                else:
                    sync_teacher_weigts = False

                self.writer.add_scalar("sync_teacher_weigts", sync_teacher_weigts, self.global_step)

                if self.cfg.colocate_all:
                    async with Timer("Backload vllm engines to gpu"):
                        await self._backload_vllm_engines()

            if self.cfg.update_ref_every_epoch and self.cfg.use_ref_model:
                if self.cfg.colocate_all:
                    await self.policy_model.backload_to_gpu()
                await self.policy_model.async_save_model(self.tokenizer, self.global_step)
                if self.cfg.colocate_all:
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

        combined_all_student_prompts, combined_all_teacher_prompts, combined_outputs, combined_custom_rewards, combined_teacher_custom_rewards, combined_answer_indices, combined_initial_scores, combined_initial_teacher_scores, combined_final_answers = [], [], [], [], [], [], [], [], []
        teacher_generated, combined_correct_formattings, combined_extras = [], [], []
        # Keep a per-prompt FIFO list of student responses and final answers to use letter for correct_incorrect augmentation if used
        student_responses_by_prompt = defaultdict(list)
        student_final_answers_by_prompt = defaultdict(list)
        student_response_ptr = defaultdict(int)
        student_correct_by_prompt = defaultdict(list)

        if self.global_step % self.cfg.generate_with_student == 0:

            n_student_samples_per_prompt = self.cfg.n_student_samples_per_prompt if self.cfg.n_student_samples_per_prompt > 0 else self.cfg.n_samples_per_prompt
            all_student_prompts = sum([[prompt[0]] * n_student_samples_per_prompt for prompt in all_inputs], [])
            all_extras = sum([[prompt[1]] * n_student_samples_per_prompt for prompt in all_inputs], [])
            # shuffle all_prompts and all_extras together
            indices = list(range(len(all_student_prompts)))
            rng = random.Random(42)
            rng.shuffle(indices)
            all_student_prompts = [all_student_prompts[i] for i in indices]
            all_extras = [all_extras[i] for i in indices]

            # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence, generate sequences via vllm engines
            async with Timer("Sync policy weights to VLLM engines for student generation"):
                # Ensure vLLM engines are configured for student
                await self._ensure_vllm_role("student")
                await self._major_sync_policy_weights_to_vllm()

            outputs = await self._distributed_generate(all_student_prompts, all_extras, teacher=False, desc="Generate student sequences via vllm engines", **generate_kwargs)

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_student_prompts) == len(outputs), f"generate objects number must be equal to all inputs number {len(all_student_prompts)} {len(outputs)}"

            # 1.2 calculate custom rewards if has custom reward function
            if self.cfg.use_compute_reward_fn:
                async with Timer("Calculate custom rewards"):
                    dp_tasks = []
                    reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                    all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers, teacher_yes, teacher_no, correct_formattings, pass_at_n_dict = await reward_fn(
                        all_student_prompts, outputs, all_extras, prefix='student/')
                    assert len(all_student_prompts) == len(outputs), "generate objects number after custom reward function must be equal to all inputs number"
            else:
                all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers, teacher_yes, teacher_no, correct_formattings, pass_at_n_dict = all_student_prompts, outputs, None, None, None, None, None, None, None, None, None, None


            for sp, sresp, sfinal, score in zip(all_student_prompts, outputs, final_answers, initial_scores):
                student_responses_by_prompt[sp].append(sresp)
                student_final_answers_by_prompt[sp].append(sfinal)
                student_correct_by_prompt[sp].append(score)

            self.buffer = [all_student_prompts, outputs, initial_scores, all_extras]

            # Log grouped student responses per prompt (captures multiple attempts)
            self.log_student_responses_by_prompt(student_responses_by_prompt, student_final_answers_by_prompt, student_correct_by_prompt, step=self.global_step)

        if self.cfg.augment_student_generation_with_teacher:

            if self.cfg.add_student_pregenerated_answers_responses:
                # Add previous student responses to the prompt for teacher generation
                r_indices = np.random.randint(0, len(all_inputs), len(all_inputs) // 4)
                for idx in r_indices:
                    if not self.buffer[2][idx]:
                        tuncated_len = np.random.randint(len(self.buffer[1])-5)
                        prompt = self.buffer[0][idx] + self.buffer[1][idx][:-tuncated_len]
                        logger.info(f"new prompt by truncated student generation: {prompt}")
                        all_inputs.append((prompt, self.buffer[3][idx]))

            n_teacher_samples_per_prompt = self.cfg.n_teacher_samples_per_prompt if self.cfg.n_teacher_samples_per_prompt > 0 else self.cfg.n_samples_per_prompt
            all_student_prompts = sum([[prompt[0]] * n_teacher_samples_per_prompt for prompt in all_inputs], [])
            all_extras = sum([[prompt[1]] * n_teacher_samples_per_prompt for prompt in all_inputs], [])

            # shuffle all_prompts and all_extras together
            indices = list(range(len(all_student_prompts)))
            rng = random.Random(42)
            rng.shuffle(indices)
            all_student_prompts = [all_student_prompts[i] for i in indices]
            all_extras = [all_extras[i] for i in indices]

            # Sync teacher model weights to VLLM engines before generation
            if self.cfg.separate_teacher_model:
                async with Timer("Sync teacher weights to VLLM engines"):
                    # Switch vLLM engines to teacher
                    await self._ensure_vllm_role("teacher")
                    await self._major_sync_teacher_weights_to_vllm()
            elif not self.cfg.generate_with_student:
                async with Timer("Sync policy weights to VLLM engines for teacher generation (there is no separate teacher model)"):
                    await self._ensure_vllm_role("student")
                    await self._major_sync_policy_weights_to_vllm()

            assert self.cfg.augment_strategy == "distill", f"Only distill augmentation strategy is supported currently, but got {self.cfg.augment_strategy}"
            all_teacher_prompts = all_student_prompts
            for extra in all_extras:
                extra["teacher_answer"] = extra["answer"]

            indices_incorrect = []
            new_indicess = np.arange(8)

            logger.info(f'len of all_teacher_prompts and all_extras: {len(all_teacher_prompts)} {len(all_extras)}')
            # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence, generate sequences via vllm engines
            outputs = await self._distributed_generate(all_teacher_prompts, all_extras, teacher=True, desc="Generate complimentary teacher sequences via vllm engines", **generate_kwargs)

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_teacher_prompts) == len(outputs), f"generate objects number must be equal to all inputs number {len(all_teacher_prompts)}, {len(outputs)}"

            # 1.2 calculate custom rewards if has custom reward function
            if self.cfg.use_compute_reward_fn:
                async with Timer("Calculate custom rewards"):
                    dp_tasks = []
                    reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                    # Use student prompts for reward calculation since that's what the model will be trained on
                    all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers, _, _, correct_formattings, _ = await reward_fn(
                        all_student_prompts, outputs, all_extras, prefix='teacher/')
                    assert len(all_student_prompts) == len(outputs) == len(
                        all_teacher_prompts), "generate objects number after custom reward function must be equal to all inputs number"
            else:
                all_student_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers, correct_formattings, pass_at_n_dict = all_student_prompts, outputs, None, None, None, None, None, None, None, None

            # Log student/teacher paired generations for the same student prompt
            self.log_paired_generation_examples(
                all_student_prompts=all_student_prompts,
                all_teacher_prompts=all_teacher_prompts,
                outputs=outputs,
                final_answers=final_answers,
                initial_teacher_scores=initial_teacher_scores,
                all_extras=all_extras,
                student_responses_by_prompt=student_responses_by_prompt,
                student_final_answers_by_prompt=student_final_answers_by_prompt,
                student_response_ptr=student_response_ptr,
                student_correct_by_prompt=student_correct_by_prompt,
                step=self.global_step,
            )

            # Log corresponding teacher generation examples to wandb
            self.log_teacher_generation_examples(
                all_student_prompts=all_student_prompts,
                all_teacher_prompts=all_teacher_prompts,
                outputs=outputs,
                final_answers=final_answers,
                initial_scores=initial_scores,
                initial_teacher_scores=initial_teacher_scores,
                indices_incorrect=indices_incorrect,
                new_indicess=new_indicess,
                step=self.global_step,
            )

            teacher_generated = [1] * len(all_student_prompts)

        # offload vllm engines when colocate all models
        if self.cfg.colocate_all:
            async with Timer("Offload vllm engines to cpu"):
                await self._offload_vllm_engines()

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
        teacher_generated = [teacher_generated[i] for i in indices]

        del final_answers

        assert self.cfg.student_loss_type in ['ppo', 'sft', 'topr'], logger.info(f"student loss type {self.cfg.student_loss_type} must be ppo, sft or topr")
        assert self.cfg.teacher_loss_type in ['ppo', 'sft', 'topr'], logger.info(f"teacher loss type {self.cfg.teacher_loss_type} must be ppo, sft or topr")

        initial_scores, initial_teacher_scores, teacher_generated = np.array(initial_scores), np.array(initial_teacher_scores), np.array(teacher_generated)
        # Fraction of teacher-generated samples (code == 1)
        self.writer.add_scalar("teacher_generated_frac", (teacher_generated == 1).mean(), self.global_step)
        logger.info(f"all_student_prompts: {len(all_student_prompts)}, all_teacher_prompts: {len(all_teacher_prompts)}")
        assert len(all_student_prompts) == len(all_teacher_prompts) == len(teacher_generated) == len(initial_scores) == len(initial_teacher_scores) == len(answer_indices) == len(outputs), (logger.info(f"student and teacher prompts must be equal in length {len(all_student_prompts)} {len(all_teacher_prompts)}"))

        # empty data
        if len(all_student_prompts) == 0:
            return

        # 1.3 packing samples
        async with Timer("Packing samples"):
            # Pack student and teacher sequences
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

        # Compute teacher reward (pre-stats); logging and GRPO normalization follow below
        (
            final_reward_list,
            kl_mean_list,
            kl_reward_list,
            window_kl_reward_list,
            kl_sum_list,
            kl_max_list,
            ss_reward_mean_list,
            ss_reward_min_list,
            ss_reward_list,
            teacher_match_reward_list,
            teacher_ratio_clipped_0_1_list,
            student_ratio_clipped_0_1_list,
            teacher_pass_at_n_dict,
            pass_at_n_dict,
        ) = await self._calculate_teacher_rewards(
            student_experiences=student_experiences,
            teacher_experiences=teacher_experiences,
            answer_indices=answer_indices,
            initial_teacher_scores=initial_teacher_scores,
            teacher_custom_rewards=teacher_custom_rewards,
            teacher_generated=teacher_generated,
            all_teacher_prompts=all_teacher_prompts,
            all_student_prompts=all_student_prompts,
            initial_scores=initial_scores,
        )

        # Log stats
        final_reward_list = np.array(final_reward_list)
        kl_mean_list = np.array(kl_mean_list)
        kl_reward_list = np.array(kl_reward_list)
        window_kl_reward_list = np.array(window_kl_reward_list)
        kl_sum_list = np.array(kl_sum_list)
        kl_max_list = np.array(kl_max_list)
        ss_reward_mean_list = np.array(ss_reward_mean_list)
        ss_reward_min_list = np.array(ss_reward_min_list)
        ss_reward_list = np.array(ss_reward_list)
        teacher_match_reward_list = np.array(teacher_match_reward_list)
        teacher_ratio_clipped_0_1_list = np.array(teacher_ratio_clipped_0_1_list)
        student_ratio_clipped_0_1_list = np.array(student_ratio_clipped_0_1_list)

        self._log_training_metrics(
            teacher_generated,
            initial_scores,
            final_reward_list,
            kl_reward_list,
            window_kl_reward_list,
            kl_mean_list,
            kl_max_list,
            kl_sum_list,
            teacher_match_reward_list,
            ss_reward_mean_list,
            ss_reward_min_list,
            ss_reward_list,
            initial_teacher_scores,
            teacher_ratio_clipped_0_1_list,
            student_ratio_clipped_0_1_list,
        )

        if self.cfg.use_grpo:
            async with Timer("computing GRPO normalized rewards"):
                await self._apply_grpo_normalization(
                    teacher_experiences,
                    student_experiences,
                    all_teacher_prompts,
                    all_student_prompts,
                    final_reward_list,
                    teacher_pass_at_n_dict,
                    pass_at_n_dict,
                    initial_scores,
                    ss_reward_list,
                )

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
