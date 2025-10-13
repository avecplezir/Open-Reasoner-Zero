import asyncio
import json
import os
import random
from functools import partial
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union
import wandb
import numpy as np
from collections import defaultdict, Counter
import gc

import torch
from loguru import logger
from tqdm import tqdm

from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    masked_mean,
    normalize_advantages,
)

from orz.ppo.base_trainer import BaseTrainer, compute_loss_type_hash


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

                # 1. eval if enable eval
                if self.cfg.enable_eval and (
                    self.global_step % self.cfg.eval_interval == 0 or iter == len(self.prompts_dataloader) - 1
                ):

                    async with Timer("Eval of the student model"):
                        await self._major_sync_policy_weights_to_vllm()
                        await self.eval(prefix="")

                    if self.cfg.separate_teacher_model and self.cfg.enable_eval and self.cfg.eval_teacher:
                        async with Timer("Eval of the teacher model"):
                            await self._major_sync_teacher_weights_to_vllm()
                            await self.eval(prefix="teacher")

                    # Optional: evaluate verifier in adversarial mode, when available
                    if self.cfg.adversarial_training and self.eval_verifier:
                        async with Timer("Eval of the verifier (adversarial)"):
                            await self.eval_verifier(prefix="verifier")

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
                    if self.cfg.student_training_rounds > 0:
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

        # Prepare BOS token for logging
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        # the same, but now generate data with student prompts
        # Create paired data (positive/negative for each prompt)
        paired_data = []
        for prompt in all_inputs:
            for _ in range(self.cfg.n_samples_per_prompt):
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
            all_extras.extend([dict(extra)])

        if self.cfg.generate_with_student:

            # Generate with student; optionally retry prompts with too few successes.
            prompt_to_extra = {p: e for p, e in all_inputs}
            original_prompts = list(set(prompt_to_extra.keys()))

            # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence, generate sequences via vllm engines
            async with Timer("Sync policy weights to VLLM engines for student generation"):
                await self._major_sync_policy_weights_to_vllm()

            outputs = await self._distributed_generate(all_student_prompts, all_extras, teacher=False, desc="Generate student sequences via vllm engines", **generate_kwargs)

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_student_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

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

            for sp, sresp, sfinal in zip(all_student_prompts, outputs, final_answers):
                student_responses_by_prompt[sp].append(sresp)
                student_final_answers_by_prompt[sp].append(sfinal)

            # create teacher prompts from student prompts
            all_teacher_prompts, indices_incorrect = self._create_teacher_prompts_from_student(all_extras, final_answers, initial_scores, initial_teacher_scores, teacher_yes, teacher_no, all_student_prompts, bos_token)

            # Log a few examples to wandb right after student generation
            self.log_student_generation_examples(
                all_student_prompts=all_student_prompts,
                all_teacher_prompts=all_teacher_prompts,
                outputs=outputs,
                final_answers=final_answers,
                initial_scores=initial_scores,
                initial_teacher_scores=initial_teacher_scores,
                indices_incorrect=indices_incorrect,
                step=self.global_step,
            )

            # Optionally skip student-generated data when configured to train only on
            # teacher-generated data for either teacher-only or student-only runs.
            if (not self.train_student and self.train_teacher and self.cfg.train_teacher_on_teacher_data_only) or (self.train_student and not self.train_teacher and self.cfg.train_student_on_teacher_data_only):
                logger.info(f"Skipping student-generated data due to train_teacher_on_teacher_data_only={self.cfg.train_teacher_on_teacher_data_only} and train_teacher_on_teacher_data_only={self.cfg.train_student_on_teacher_data_only} flags")
            else:
                # Remember where the original student-generated block starts so we can
                # attach student-side match rewards later (after adversarial gen)
                teacher_generated.extend([0] * len(all_student_prompts))
                combined_all_student_prompts.extend(all_student_prompts)
                combined_all_teacher_prompts.extend(all_teacher_prompts)
                combined_outputs.extend(outputs)
                combined_custom_rewards.extend(custom_rewards)
                combined_teacher_custom_rewards.extend(teacher_custom_rewards)
                combined_answer_indices.extend(answer_indices)
                combined_initial_scores.extend(initial_scores)
                combined_initial_teacher_scores.extend(initial_teacher_scores)
                combined_final_answers.extend(final_answers)
                combined_correct_formattings.extend(correct_formattings)
                combined_extras.extend(all_extras)

            # Optional retry rounds based on per-prompt success counts (pass@N)
            await self._retry_student_generation_pass_at_n(
                original_prompts=original_prompts,
                prompt_to_extra=prompt_to_extra,
                pass_at_n_dict=pass_at_n_dict,
                bos_token=bos_token,
                generate_kwargs=generate_kwargs,
                extras_for_append=all_extras,
                combined_all_student_prompts=combined_all_student_prompts,
                combined_all_teacher_prompts=combined_all_teacher_prompts,
                combined_outputs=combined_outputs,
                combined_custom_rewards=combined_custom_rewards,
                combined_teacher_custom_rewards=combined_teacher_custom_rewards,
                combined_answer_indices=combined_answer_indices,
                combined_initial_scores=combined_initial_scores,
                combined_initial_teacher_scores=combined_initial_teacher_scores,
                combined_final_answers=combined_final_answers,
                combined_correct_formattings=combined_correct_formattings,
                combined_extras=combined_extras,
                teacher_generated=teacher_generated,
            )
        else:
            final_answers = initial_scores = initial_teacher_scores = teacher_yes = teacher_no = []


        generate_with_teacher = not (self.train_teacher and self.cfg.train_teacher_on_student_data_only)
        if generate_with_teacher:
            logger.info("Skipping teacher generation since only training teacher on student data")

        if generate_with_teacher and self.cfg.augment_student_generation_with_teacher:

            # Sync teacher model weights to VLLM engines before generation
            if self.cfg.separate_teacher_model:
                async with Timer("Sync teacher weights to VLLM engines"):
                    await self._major_sync_teacher_weights_to_vllm()
            elif not self.cfg.generate_with_student:
                async with Timer("Sync policy weights to VLLM engines for teacher generation (there is no separate teacher model)"):
                    await self._major_sync_policy_weights_to_vllm()

            # create the complementary teacher prompt(s) and collect data with it
            all_teacher_prompts, all_student_prompts, aug_all_extras, indices_incorrect, new_indicess = self._augment_student_generation_with_teacher(all_student_prompts, all_extras, final_answers, initial_scores, initial_teacher_scores, teacher_yes, teacher_no, bos_token)
            logger.info(f"extras, double extras, and augmented extras lengths, {len(all_extras)} {2 * len(all_extras)} {len(aug_all_extras)}")
            all_extras = aug_all_extras

            # 1. generate sequences and inference, calculate values, log probs, rewards, kl divergence, generate sequences via vllm engines
            outputs = await self._distributed_generate(all_teacher_prompts, all_extras, teacher=True, desc="Generate complimentary teacher sequences via vllm engines", **generate_kwargs)

            # skip when data is not enough
            if len(outputs) <= 0:
                return

            assert len(all_teacher_prompts) == len(outputs), "generate objects number must be equal to all inputs number"

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

            teacher_generated.extend([1] * len(all_student_prompts))
            combined_all_student_prompts.extend(all_student_prompts)
            combined_all_teacher_prompts.extend(all_teacher_prompts)
            combined_outputs.extend(outputs)
            combined_custom_rewards.extend(custom_rewards)
            combined_teacher_custom_rewards.extend(teacher_custom_rewards)
            combined_answer_indices.extend(answer_indices)
            combined_initial_scores.extend(initial_scores)
            combined_initial_teacher_scores.extend(initial_teacher_scores)
            combined_final_answers.extend(final_answers)
            combined_correct_formattings.extend(correct_formattings)
            combined_extras.extend(all_extras)

        # Optional verification round generation on adversarial teacher or helping student samples
        if self.cfg.adversarial_training:

            async with Timer("Generating verification responses"):
                adv_prompts, adv_init_prompts, adv_extras, adv_teacher_index_groups = self._build_adversarial_student_prompts(
                    combined_outputs,
                    combined_all_teacher_prompts,
                    combined_all_student_prompts,
                    combined_extras,
                    teacher_generated,
                    bos_token,
                    combined_final_answers,
                )

            if self.cfg.separate_teacher_model:
                # Sync student weights and generate adversarial student responses
                async with Timer("Sync policy weights to VLLM engines for adversarial student gen"):
                    await self._major_sync_policy_weights_to_vllm()

            adv_outputs_local: List[str] = await self._distributed_generate(
                adv_prompts,
                adv_extras,
                teacher=False,
                desc="Generate adversarial student sequences via vllm engines",
                **generate_kwargs,
            )

            reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
            (
                adv_prompts,
                adv_outputs,
                adv_custom_rewards,
                adv_teacher_custom_rewards,
                adv_answer_indices,
                adv_initial_scores,
                adv_initial_teacher_scores,
                adv_final_answers,
                adv_teacher_yes,
                adv_teacher_no,
                adv_correct_formattings,
                adv_pass_at_n_dict,
            ) = await reward_fn(adv_prompts, adv_outputs_local, adv_extras, prefix='adv_student/')

            # teacher_adv_match_rewards = []
            # If we have generated adversarial responses for each teacher prompt, compute
            # teacher rewards as the average agreement of adversarial final answers with
            # the teacher-declared answer embedded in the prompts. This averages over
            # multiple adversarial responses per teacher prompt instance.
            # Each teacher prompt instance produced cfg.n_samples_per_prompt adversarial responses
            logger.info(f"Generated {len(adv_prompts)} verifier responses for {len(adv_teacher_index_groups)} adversarial groups")
            assert len(adv_initial_teacher_scores) == len(adv_teacher_index_groups) * self.cfg.adv_n_samples_per_prompt, (
                "Expected adv_initial_teacher_scores to be groups * n_samples_per_prompt"
            )
            # Overwrite the teacher custom rewards block we appended earlier
            index_group_dict = {}
            for g in range(len(adv_teacher_index_groups)):
                start = g * self.cfg.adv_n_samples_per_prompt
                end = (g + 1) * self.cfg.adv_n_samples_per_prompt
                avg_teacher_match = float(np.mean(adv_initial_teacher_scores[start:end]))
                index_group_dict[adv_prompts[start]] = adv_teacher_index_groups[g]

                # teacher_adv_match_rewards.append(avg_teacher_match)

                # Assign teacher reward to all indices participating in this mixed group
                for pos, idx in enumerate(adv_teacher_index_groups[g]):
                    if len(adv_teacher_index_groups[g]) == 1:
                        # Single-teacher groups get the direct average match reward
                        combined_teacher_custom_rewards[idx][-1] = avg_teacher_match
                    else:
                        if pos == 0:
                            # Mixed group, first index is "yes"
                            combined_teacher_custom_rewards[idx][-1] = avg_teacher_match
                        elif pos == 1:
                            # Mixed group, second index is "no"
                            combined_teacher_custom_rewards[idx][-1] = 1.0 - avg_teacher_match
                        else:
                            assert False, "Only support mixed groups of size 2 for now"

                # Compute student-side adversarial match average for this group
                avg_student_match = float(np.mean(adv_initial_scores[start:end]))
                # Apply negative strategy separately per teacher index using its original correctness
                for idx in adv_teacher_index_groups[g]:
                    adj_student_match = avg_student_match
                    if not combined_initial_scores[idx]:
                        if self.cfg.avd_student_negative_strategy == "inverse":
                            adj_student_match = 1 - adj_student_match
                        elif self.cfg.avd_student_negative_strategy == "negate":
                            adj_student_match = -adj_student_match
                        elif self.cfg.avd_student_negative_strategy == "inv_neg":
                            adj_student_match = -(1 - adj_student_match)
                        # "same" leaves it unchanged

                    if self.cfg.adv_student_add_initial:
                        adj_student_match = adj_student_match + combined_custom_rewards[idx][-1]

                    if self.train_student:
                        combined_custom_rewards[idx][-1] = adj_student_match

            for g in range(len(adv_teacher_index_groups)):
                idx1, idx2 = adv_teacher_index_groups[g]
                two_index_sum = combined_teacher_custom_rewards[idx1][-1] + combined_teacher_custom_rewards[idx2][-1]
                assert 0.99 <= two_index_sum <= 1.01, f"Sum of teacher rewards for mixed group must be 1, got {two_index_sum}"

            self.log_adversarial_examples(
                student_prompts=combined_all_student_prompts,
                teacher_prompts=combined_all_teacher_prompts,
                combined_custom_rewards=combined_custom_rewards,
                combined_teacher_custom_rewards=combined_teacher_custom_rewards,
                index_group_dict=index_group_dict,
                adv_prompts=adv_prompts,
                adv_outputs=adv_outputs,
                adv_final_answers=adv_final_answers,
                adv_extras=adv_extras,
                adv_initial_scores=adv_initial_scores,
                adv_initial_teacher_scores=adv_initial_teacher_scores,
                step=self.global_step,
            )

            # Use adv_prompts as both student and teacher prompts for these appended samples
            teacher_generated.extend([-1] * len(adv_prompts))
            combined_all_student_prompts.extend(adv_prompts)
            combined_all_teacher_prompts.extend(adv_prompts)
            combined_outputs.extend(adv_outputs)
            combined_custom_rewards.extend(adv_custom_rewards)
            combined_teacher_custom_rewards.extend(adv_teacher_custom_rewards)
            combined_answer_indices.extend(adv_answer_indices)
            combined_initial_scores.extend(adv_initial_scores)
            combined_initial_teacher_scores.extend(adv_initial_teacher_scores)
            combined_final_answers.extend(adv_final_answers)
            combined_correct_formattings.extend(adv_correct_formattings)
            combined_extras.extend(adv_extras)

        # offload vllm engines when colocate all models
        if self.cfg.colocate_all:
            async with Timer("Offload vllm engines to cpu"):
                await self._offload_vllm_engines()

        all_student_prompts, all_teacher_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, final_answers, correct_formattings = \
            combined_all_student_prompts, combined_all_teacher_prompts, combined_outputs, combined_custom_rewards, combined_teacher_custom_rewards, combined_answer_indices, combined_initial_scores, combined_initial_teacher_scores, combined_final_answers, combined_correct_formattings

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
        correct_formattings = [correct_formattings[i] for i in indices]

        del final_answers

        assert self.cfg.student_loss_type in ['ppo', 'sft', 'topr'], logger.info(f"student loss type {self.cfg.student_loss_type} must be ppo, sft or topr")
        assert self.cfg.teacher_loss_type in ['ppo', 'sft', 'topr'], logger.info(f"teacher loss type {self.cfg.teacher_loss_type} must be ppo, sft or topr")

        all_student_prompts, all_teacher_prompts, outputs, custom_rewards, teacher_custom_rewards, answer_indices, initial_scores, initial_teacher_scores, teacher_generated, correct_formattings  = self._filter_samples_for_training(
            all_student_prompts,
            all_teacher_prompts,
            outputs,
            custom_rewards,
            teacher_custom_rewards,
            answer_indices,
            initial_scores,
            initial_teacher_scores,
            teacher_generated,
            correct_formattings,
        )

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
