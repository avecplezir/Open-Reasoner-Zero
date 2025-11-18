import os
import re
import math
import asyncio
import random
from functools import partial
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional, Callable, Awaitable

import numpy as np
import torch
import torch.nn.functional as F
import ray
from loguru import logger
import wandb
from omegaconf.dictconfig import DictConfig
from orz.ppo.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from torch.utils.data import DataLoader
from orz.ppo.dataset import BalancedYesNoBatchSampler
from torch.utils.tensorboard import SummaryWriter
from ray.util.placement_group import PlacementGroup, placement_group

from orz.ppo.actors import PPORayActorGroup
from orz.ppo.replay_buffer import Experience, NaiveReplayBuffer
from orz.ppo.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
)
from playground.zero_setting_base import (
    create_teacher_prompt_from_answer,
)

class BaseTrainer:
    """
    Base trainer with shared helpers that are stable and reusable across trainers.
    Contains small utilities and prompt-building helpers extracted from the
    make_experience flow to keep the main trainer lean.
    """

    def __init__(
        self,
        cfg: DictConfig,
        strategy: DeepspeedStrategy,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        vllm_engines=None,
        colocate_pg: Optional[PlacementGroup] = None,
        vllm_pg_handles: Optional[list] = None,
        teacher_vllm_engines=None,
    ):
        if vllm_pg_handles is None:
            vllm_pg_handles = []
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.student_vllm_engines = vllm_engines
        self.teacher_vllm_engines = teacher_vllm_engines
        self.vllm_engines = vllm_engines
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg
        self._dual_vllm_enabled = (
            self.cfg.colocate_all
            and self.cfg.separate_teacher_model
            and self.cfg.separate_teacher_vllm_engine
            and self.teacher_vllm_engines is not None
        )
        logger.info(f"_dual_vllm_enabled: {self._dual_vllm_enabled}")
        if self.teacher_vllm_engines is not None and not self._dual_vllm_enabled:
            raise ValueError("teacher_vllm_engines provided but dual VLLM support is not enabled in config")
        # Track which role the current vLLM engines are set up for.
        self._vllm_current_role: Optional[str] = "student"
        # self._vllm_pg_handles = vllm_pg_handles

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

    # Provide YES/NO tokens based on current configuration
    def yes_token(self) -> str:
        return "\\boxed{yes}" if self.cfg.boxed_pattern else "yes"

    def no_token(self) -> str:
        return "\\boxed{no}" if self.cfg.boxed_pattern else "no"

    async def eval(self):
        raise NotImplementedError("Eval function should be implemented in user's exp")

    def _log_wandb_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ) -> None:
        """
        Helper to log a W&B table in a consistent way.
        - Skips when W&B is not initialized or data is empty.
        - Uses current global step if step is not provided.
        """
        try:
            if wandb.run is None:
                return
        except Exception:
            # wandb not initialized or unavailable
            return

        if not data:
            return

        try:
            wandb.log(
                {name: wandb.Table(columns=columns, data=data)},
                step=self.global_step if step is None else step,
            )
        except Exception as e:
            logger.warning(f"Failed to log W&B table '{name}': {e}")

    def _create_teacher_prompts_from_student(
        self,
        all_extras: List[dict],
        final_answers: List[str],
        initial_scores: List[bool],
        initial_teacher_scores: List[bool],
        teacher_yes: List[bool],
        teacher_no: List[bool],
        all_student_prompts: List[str],
        bos_token: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Build teacher prompts from student outputs. Also annotates extras with
        the chosen teacher_answer. Returns the constructed teacher prompts and
        indices of incorrect student samples for logging.
        """
        all_teacher_prompts: List[str] = []
        indices_incorrect: List[int] = []

        for i, (extra, final_answer, student_score, teacher_score) in enumerate(
            zip(all_extras, final_answers, initial_scores, initial_teacher_scores)
        ):
            if teacher_score:
                if teacher_yes[i]:
                    student_answer = self.yes_token()
                elif teacher_no[i]:
                    student_answer = self.no_token()
                else:
                    assert False, f"final_answer {final_answer} must be yes or no"
            else:
                student_answer = final_answer

            teacher_prompt = create_teacher_prompt_from_answer(
                extra["dialogue"],
                student_answer,
                bos_token,
                cfg=self.cfg,
                is_correct=bool(student_score),
            )

            all_teacher_prompts.append(teacher_prompt)
            extra["teacher_answer"] = student_answer
            if not student_score:
                indices_incorrect.append(i)

        assert len(all_student_prompts) == len(all_teacher_prompts), (
            "student and teacher prompts must be equal in length"
        )
        return all_teacher_prompts, indices_incorrect

    def log_student_responses_by_prompt(
        self,
        student_responses_by_prompt,
        student_final_answers_by_prompt,
        student_correct_by_prompt,
        max_prompts: int = 3,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a W&B table aggregating all student responses for each unique prompt.
        Creates one row per (prompt, response_idx) so multiple responses for the
        same prompt are captured and queryable.
        """
        if not student_responses_by_prompt:
            return

        rows: List[List[Any]] = []
        # Limit the number of prompts to avoid oversized tables each step
        prompts = list(student_responses_by_prompt.keys())[:max_prompts]
        for p in prompts:
            responses = student_responses_by_prompt.get(p, [])

            for idx, resp in enumerate(responses):
                rows.append([p, idx, resp, student_final_answers_by_prompt[p][idx], student_correct_by_prompt[p][idx]])

        self._log_wandb_table(
            name="student_responses_by_prompt",
            columns=[
                "prompt",
                "response_idx",
                "student_response",
                "student_final_answer",
                "correct",
            ],
            data=rows,
            step=step,
        )

    def log_paired_generation_examples(
        self,
        *,
        all_student_prompts: List[str],
        all_teacher_prompts: List[str],
        outputs: List[Any],
        final_answers: List[Any],
        initial_teacher_scores: List[Any],
        all_extras: List[dict],
        student_responses_by_prompt: Dict[str, List[str]],
        student_final_answers_by_prompt: Dict[str, List[str]],
        student_response_ptr: Dict[str, int],
        student_correct_by_prompt=None,
        step: Optional[int] = None,
    ) -> None:

        paired_table_correct = []
        paired_table_incorrect = []

        if len(student_responses_by_prompt) == 0:
            return

        for i in range(len(all_teacher_prompts)):
            s_prompt = all_student_prompts[i]
            t_prompt = all_teacher_prompts[i]
            t_resp = outputs[i]
            t_final = final_answers[i]

            s_list = student_responses_by_prompt.get(s_prompt, [])
            if len(s_list) == 0:
                continue
            s_final_list = student_final_answers_by_prompt.get(s_prompt, [])
            student_correct_list = student_correct_by_prompt.get(s_prompt, [])
            idx = student_response_ptr.get(s_prompt, 0)
            s_resp = s_list[idx]
            s_final = s_final_list[idx]
            s_correct = student_correct_list[idx]

            student_response_ptr[s_prompt] = idx + 1

            correct_answer = all_extras[i].get("answer", "")
            t_correct = initial_teacher_scores[i]

            if t_correct:
                if s_correct and len(paired_table_correct) < 20:
                    paired_table_correct.append([
                        s_prompt,
                        s_resp,
                        s_final,
                        t_prompt,
                        t_resp,
                        t_final,
                        correct_answer,
                    ])
                elif not s_correct and len(paired_table_incorrect) < 20:
                    paired_table_incorrect.append([
                        s_prompt,
                        s_resp,
                        s_final,
                        t_prompt,
                        t_resp,
                        t_final,
                        correct_answer,
                    ])

            if len(paired_table_correct) >= 20 and len(paired_table_incorrect) >= 20:
                break

        self._log_wandb_table(
            name="paired_correct_examples",
            columns=[
                "student_prompt",
                "student_response",
                "student_final_answer",
                "teacher_prompt",
                "teacher_response",
                "teacher_final_answer",
                "correct_answer",
            ],
            data=paired_table_correct,
            step=step,
        )

        self._log_wandb_table(
            name="paired_incorrect_examples",
            columns=[
                "student_prompt",
                "student_response",
                "student_final_answer",
                "teacher_prompt",
                "teacher_response",
                "teacher_final_answer",
                "correct_answer",
            ],
            data=paired_table_incorrect,
            step=step,
        )

    def log_teacher_generation_examples(
        self,
        *,
        all_student_prompts: List[str],
        all_teacher_prompts: List[str],
        outputs: List[Any],
        final_answers: List[Any],
        initial_scores: List[Any],
        initial_teacher_scores: List[Any],
        indices_incorrect: List[int],
        new_indicess: List[int],
        step: Optional[int] = None,
    ) -> None:
        table_data: List[List[Any]] = []
        n = min(5, len(all_teacher_prompts))
        for i in range(n):
            idx = new_indicess[i]
            table_data.append([
                all_student_prompts[idx],
                all_teacher_prompts[idx],
                outputs[idx],
                final_answers[idx],
                bool(initial_scores[idx]),
                bool(initial_teacher_scores[idx]),
            ])
        if self.cfg.augment_strategy != "only_wrong":
            n_inc = min(5, len(indices_incorrect))
            for i in range(n_inc):
                idx = indices_incorrect[i]
                table_data.append([
                    all_student_prompts[idx],
                    all_teacher_prompts[idx],
                    outputs[idx],
                    final_answers[idx],
                    bool(initial_scores[idx]),
                    bool(initial_teacher_scores[idx]),
                ])
        self._log_wandb_table(
            name="teacher_generation_examples",
            columns=[
                "student_prompt",
                "teacher_prompt",
                "output",
                "final_answer",
                "student_correct",
                "teacher_correct",
            ],
            data=table_data,
            step=step,
        )

    # Removed: adversarial example logging

    def log_reasoning_examples(
        self,
        *,
        prompts: List[str],
        outputs: List[dict],
        extras: List[dict],
        step: Optional[int] = None,
        name: str = "reasoning_examples",
    ) -> None:
        examples = []
        for i in range(min(5, len(outputs))):
            true_answer = extras[i].get("answer", "N/A") if i < len(extras) else "N/A"
            ex = outputs[i]
            examples.append([
                prompts[i],
                ex.get("response", ""),
                ex.get("final_answer", ""),
                true_answer,
                ex.get("iscorrect", False),
                ex.get("teacher_iscorrect", False),
                ex.get("stop_reason", ""),
            ])
        self._log_wandb_table(
            name=name,
            columns=[
                "prompt",
                "reasoning_chain",
                "generated_answer",
                "true_answer",
                "is_correct",
                "teacher_iscorrect",
                "stop_reason",
            ],
            data=examples,
            step=step,
        )

    def log_eval_examples(
        self,
        *,
        output_for_save: List[dict],
        name: str,
        step: Optional[int] = None,
    ) -> None:
        examples = []
        for item in output_for_save[:5]:
            examples.append([
                item.get("prompt", ""),
                item.get("output", ""),
                item.get("final_answer", ""),
                item.get("answer", ""),
                item.get("iscorrect", False),
            ])
        self._log_wandb_table(
            name=name,
            columns=[
                "prompt",
                "reasoning_chain",
                "generated_answer",
                "true_answer",
                "is_correct",
            ],
            data=examples,
            step=step,
        )

    async def _distributed_generate(
        self,
        prompts: List[str],
        extras: List[dict] | None,
        *,
        teacher: bool,
        desc: str,
        **generate_kwargs,
    ) -> List[str]:
        """Shard prompts across vLLM engines, generate, and flatten outputs."""
        outputs: List[str] = []
        num_vllm_dp_gruops = len(self.vllm_engines)
        async with Timer(desc):
            dp_prompt_size = (len(prompts) + num_vllm_dp_gruops - 1) // num_vllm_dp_gruops
            logger.info(f"dp_prompt_size: {dp_prompt_size}")
            dp_tasks = []
            for dp_rank in range(num_vllm_dp_gruops):
                dp_inputs = prompts[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                dp_extras = None if extras is None else extras[dp_rank * dp_prompt_size : (dp_rank + 1) * dp_prompt_size]
                if len(dp_inputs) == 0:
                    logger.info(f'len of dp_inputs is 0!')
                    continue
                gen_func = self._get_generate_function(dp_rank)
                dp_tasks.append(
                    self.generate_vllm(gen_func, dp_inputs, extras=dp_extras, teacher=teacher, **generate_kwargs)
                )

            logger.info("start generation from prompts")
            local_responses = await asyncio.gather(*dp_tasks)
            outputs.extend(sum(local_responses, []))
            logger.info("generate local rollout batch done")
        return outputs

    async def _retry_student_generation_pass_at_n(
        self,
        original_prompts: List[str],
        prompt_to_extra: Dict[str, dict],
        pass_at_n_dict: Dict[str, List[int]],
        bos_token: str,
        generate_kwargs: Dict[str, Any],
        extras_for_append: List[dict],
        combined_all_student_prompts: List[str],
        combined_all_teacher_prompts: List[str],
        combined_outputs: List[str],
        combined_custom_rewards: List[Any],
        combined_teacher_custom_rewards: List[Any],
        combined_answer_indices: List[Any],
        combined_initial_scores: List[Any],
        combined_initial_teacher_scores: List[Any],
        combined_final_answers: List[Any],
        combined_correct_formattings: List[Any],
        combined_extras: List[dict],
        teacher_generated: List[bool],
        **generate_kwargs_passthrough,
    ) -> None:
        """
        Optional retry rounds based on per-prompt success counts (pass@N).
        Mutates the provided combined_* containers by appending new retry data.
        """
        if self.cfg.student_retry_max_rounds <= 0 or self.cfg.student_success_min_per_prompt <= 0:
            return

        # Initialize success counts from the first round
        success_counts = defaultdict(int)
        for p in original_prompts:
            success_counts[p] = int(sum(pass_at_n_dict.get(p, [])))

        # Build list of prompts to retry
        to_retry_keys = [p for p in original_prompts if success_counts[p] < self.cfg.student_success_min_per_prompt]
        current_prompts = [(p, prompt_to_extra[p]) for p in to_retry_keys]

        for round_idx in range(self.cfg.student_retry_max_rounds):
            if len(current_prompts) == 0:
                break

            logger.info(f'round {round_idx} with {len(to_retry_keys)} keys')

            # Duplicate prompts for this round
            paired_data = []
            for p, e in current_prompts:
                for _ in range(self.cfg.n_samples_per_prompt):
                    paired_data.append((p, e))

            rng = random.Random(4242 + round_idx)
            rng.shuffle(paired_data)

            retry_student_prompts = [p for p, _ in paired_data]
            retry_extras = [e for _, e in paired_data]

            # Generate sequences via vLLM
            retry_outputs: List[str] = await self._distributed_generate(
                retry_student_prompts,
                retry_extras,
                teacher=False,
                desc=f"Generate retry student sequences via vllm engines (round {round_idx + 1})",
                **generate_kwargs,
            )

            if len(retry_outputs) == 0:
                break

            # Score retries
            reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
            (
                retry_student_prompts,
                retry_outputs,
                retry_custom_rewards,
                retry_teacher_custom_rewards,
                retry_answer_indices,
                retry_initial_scores,
                retry_initial_teacher_scores,
                retry_final_answers,
                retry_teacher_yes,
                retry_teacher_no,
                retry_correct_formattings,
                retry_pass_at_n_dict,
            ) = await reward_fn(retry_student_prompts, retry_outputs, retry_extras, prefix=f'student_{round_idx+2}/')

            # Update counts and decide next retries
            for p, vals in retry_pass_at_n_dict.items():
                success_counts[p] += int(sum(vals))

            # Create teacher prompts for retries (for parity/logging)
            retry_teacher_prompts: List[str] = []
            for i, (extra, student_score, teacher_score) in enumerate(
                zip(retry_extras, retry_initial_scores, retry_initial_teacher_scores)
            ):
                if teacher_score:
                    if retry_teacher_yes[i]:
                        student_answer = self.yes_token()
                    elif retry_teacher_no[i]:
                        student_answer = self.no_token()
                    else:
                        assert False, "final_answer must be yes or no"
                else:
                    student_answer = self.yes_token() if random.random() > 0.5 else self.no_token()

                teacher_prompt = create_teacher_prompt_from_answer(
                    extra["dialogue"],
                    student_answer,
                    bos_token,
                    cfg=self.cfg,
                    is_correct=bool(student_score),
                )
                retry_teacher_prompts.append(teacher_prompt)

            # Append to combined containers
            teacher_generated.extend([0] * len(retry_student_prompts))
            combined_all_student_prompts.extend(retry_student_prompts)
            combined_all_teacher_prompts.extend(retry_teacher_prompts)
            combined_outputs.extend(retry_outputs)
            combined_custom_rewards.extend(retry_custom_rewards)
            combined_teacher_custom_rewards.extend(retry_teacher_custom_rewards)
            combined_answer_indices.extend(retry_answer_indices)
            combined_initial_scores.extend(retry_initial_scores)
            combined_initial_teacher_scores.extend(retry_initial_teacher_scores)
            combined_final_answers.extend(retry_final_answers)
            combined_correct_formattings.extend(retry_correct_formattings)
            combined_extras.extend(extras_for_append)

            # Prepare next-round retry list
            to_retry_keys = [
                p for p in original_prompts if success_counts[p] < self.cfg.student_success_min_per_prompt
            ]
            current_prompts = [(p, prompt_to_extra[p]) for p in to_retry_keys]

            pass_at_n_retry = sum(1 for v in success_counts.values() if np.sum(v) > 0) / max(
                1, len(pass_at_n_dict)
            )
            logger.info(f"pass_at_n_retry {round_idx} {pass_at_n_retry}")
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_scalar(f"pass_at_n_retry_{round_idx+2}", pass_at_n_retry, self.global_step)


    async def _apply_grpo_normalization(
        self,
        teacher_experiences,
        student_experiences,
        all_teacher_prompts,
        all_student_prompts,
        final_reward_list,
        teacher_pass_at_n_dict,
        pass_at_n_dict,
        initial_scores,
        ss_reward_list,
    ) -> None:
        """
        Apply GRPO normalization to teacher and student custom rewards in-place.
        Logs summary stats to TensorBoard and info logs.
        """
        prompt_idx = 0
        teacher_score_sum = 0
        score_sum = 0
        for teacher_exp, student_exp in zip(teacher_experiences, student_experiences):
            assert len(teacher_exp.info['custom_rewards']) == len(teacher_exp.num_actions[0]), (
                "teacher_exp.info['custom_rewards'] must be equal to teacher_exp.num_actions[0]"
            )
            for i in range(len(teacher_exp.num_actions[0])):
                # Teacher score
                if self.train_teacher and self.cfg.teacher_loss_type == 'sft':
                    teacher_score = 1
                else:
                    prompt = all_teacher_prompts[prompt_idx]
                    teacher_score = final_reward_list[prompt_idx].item()
                    teacher_score -= np.mean(teacher_pass_at_n_dict[prompt])
                    if not self.cfg.teacher_turn_off_std_normalization:
                        if teacher_std := np.std(teacher_pass_at_n_dict[prompt]) > 0:
                            teacher_score /= teacher_std

                teacher_score_sum += teacher_score
                teacher_exp.info['custom_rewards'][i][-1] = teacher_score

                # Student score
                if self.cfg.student_loss_type == 'sft':
                    score = initial_scores[prompt_idx]
                elif self.cfg.remove_student_reward_normalization:
                    if self.cfg.use_ss_reward_for_student:
                        signed = 1.0 if initial_scores[prompt_idx] == 1 else -1.0
                        score = float(np.exp(ss_reward_list[prompt_idx]) * signed)
                    else:
                        score = 1.0 if initial_scores[prompt_idx] == 1 else -1.0
                else:
                    if self.cfg.use_ss_reward_for_student:
                        prompt = all_student_prompts[prompt_idx]
                        signed = 1.0 if initial_scores[prompt_idx] == 1 else -1.0
                        score = float(np.exp(ss_reward_list[prompt_idx]) * signed)
                        score -= np.mean(pass_at_n_dict[prompt])
                        if std := np.std(pass_at_n_dict[prompt]) > 0:
                            score /= std
                    else:
                        prompt = all_student_prompts[prompt_idx]
                        score = float(initial_scores[prompt_idx])
                        score -= np.mean(pass_at_n_dict[prompt])
                        if std := np.std(pass_at_n_dict[prompt]) > 0:
                            score /= std

                student_exp.info['custom_rewards'][i][-1] = score
                score_sum += score

                prompt_idx += 1

        assert prompt_idx == len(all_teacher_prompts) == len(final_reward_list), (
            "last teacher prompt idx must be equal to all teacher prompts length"
        )

        # Log attempt length distributions
        logger.info(f"sum student {sum([len(v) for v in pass_at_n_dict.values()])}, sum teacher {sum([len(v) for v in teacher_pass_at_n_dict.values()])}")
        pass_n_len_dist = dict(Counter(len(v) for v in pass_at_n_dict.values()))
        teacher_pass_n_len_dist = dict(Counter(len(v) for v in teacher_pass_at_n_dict.values()))
        logger.info(f"pass_at_n attempt lengths distribution: {pass_n_len_dist}")
        logger.info(f"teacher_pass_at_n attempt lengths distribution: {teacher_pass_n_len_dist}")

        avg_pass_at_n = (sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / max(1, len(pass_at_n_dict)))
        self.writer.add_scalar("avg_teacher_reward_normalized", teacher_score_sum / max(1, len(all_teacher_prompts)), self.global_step)
        self.writer.add_scalar("avg_student_reward_normalized", score_sum / max(1, len(all_student_prompts)), self.global_step)
        self.writer.add_scalar("avg_pass_at_n_combined", avg_pass_at_n, self.global_step)
        logger.info(f"avg_teacher_reward: {teacher_score_sum / max(1, len(all_teacher_prompts))}, avg_student_reward: {score_sum / max(1, len(all_student_prompts))}, avg_pass_at_n: {avg_pass_at_n}")

    def _log_training_metrics(
        self,
        teacher_generated: np.ndarray,
        initial_scores: np.ndarray,
        final_reward_list: np.ndarray,
        kl_reward_list: np.ndarray,
        window_kl_reward_list: np.ndarray,
        kl_mean_list: np.ndarray,
        kl_max_list: np.ndarray,
        kl_sum_list: np.ndarray,
        teacher_match_reward_list: np.ndarray,
        ss_reward_mean_list: np.ndarray,
        ss_reward_min_list: np.ndarray,
        ss_reward_list: np.ndarray,
        initial_teacher_scores: np.ndarray,
        teacher_ratio_clipped_0_1_list: np.ndarray,
        student_ratio_clipped_0_1_list: np.ndarray,
    ) -> None:

        kl_ss_reward_ratio = np.clip(kl_reward_list / ss_reward_list, a_min=None, a_max=1)

        log_dict: Dict[str, float] = {}
        for prefix in ["", "teacher", "student", "adv"]:
            if prefix == "teacher":
                mask = teacher_generated == 1
            elif prefix == "student":
                mask = teacher_generated == 0
            elif prefix == "adv":
                mask = teacher_generated == -1
            else:
                # Exclude adversarial (-1) from aggregate slice
                mask = (teacher_generated == 1) | (teacher_generated == 0)

            # Stop logging for adv when there are no samples
            if prefix == "adv" and mask.sum() == 0:
                continue

            logger.info(f"{prefix} slice {mask.sum()} samples")

            avg_kl_ss_reward_ratio = kl_ss_reward_ratio[mask].mean()
            avg_student_reward = initial_scores[mask].mean()
            avg_teacher_reward = final_reward_list[mask].mean()
            avg_student_teacher_kl = kl_reward_list[mask].mean()
            avg_window_kl = window_kl_reward_list[mask].mean() if len(window_kl_reward_list) > 0 else 0
            avg_student_teacher_kl_mean = kl_mean_list[mask].mean()
            avg_student_teacher_kl_max = kl_max_list[mask].mean()
            avg_teacher_match_reward = teacher_match_reward_list[mask].mean()
            avg_ss_reward_mean = ss_reward_mean_list[mask].mean()
            avg_ss_reward_min = ss_reward_min_list[mask].mean()
            avg_ss_reward = ss_reward_list[mask].mean()

            ct = np.logical_and(initial_scores == 1, mask)
            it = np.logical_and(initial_scores == 0, mask)

            correct_match_reward_trainer = np.array([]) if np.all(it) else np.array(teacher_match_reward_list[ct])
            incorrect_match_reward_trainer = np.array([]) if np.all(ct) else np.array(teacher_match_reward_list[it])
            avg_teacher_correct_match_reward = 0 if len(correct_match_reward_trainer) == 0 else np.mean(correct_match_reward_trainer).item()
            avg_teacher_incorrect_match_reward = 0 if len(incorrect_match_reward_trainer) == 0 else np.mean(incorrect_match_reward_trainer).item()

            ic = np.logical_and(np.logical_and(initial_scores == 0, initial_teacher_scores == 1), mask)
            cc = np.logical_and(np.logical_and(initial_scores == 1, initial_teacher_scores == 1), mask)
            ii = np.logical_and(np.logical_and(initial_scores == 0, initial_teacher_scores == 0), mask)

            logger.info(f"{prefix} {ic.mean()} ic, {cc.mean()} cc, {ii.mean()} ii")

            avg_correct_kl_sum = kl_sum_list[cc].mean()
            avg_incorrect_kl_sum = kl_sum_list[ic].mean()
            avg_correct_kl_mean = kl_mean_list[cc].mean()
            avg_incorrect_kl_mean = kl_mean_list[ic].mean()
            avg_correct_kl_max = kl_max_list[cc].mean()
            avg_incorrect_kl_max = kl_max_list[ic].mean()
            avg_correct_ss_reward_mean = ss_reward_mean_list[cc].mean()
            avg_incorrect_ss_reward_mean = ss_reward_mean_list[ic].mean()
            avg_correct_ss_reward_min = ss_reward_min_list[cc].mean()
            avg_incorrect_ss_reward_min = ss_reward_min_list[ic].mean()
            teacher_correct_ratio_clipped_0_1 = (
                np.array([])
                if np.all(ic) or not self.cfg.teacher_loss_type == 'topr'
                else np.array(teacher_ratio_clipped_0_1_list[cc])
            )
            teacher_incorrect_ratio_clipped_0_1 = (
                np.array([])
                if np.all(cc) or not self.cfg.teacher_loss_type == 'topr'
                else np.array(teacher_ratio_clipped_0_1_list[ic])
            )
            student_correct_ratio_clipped_0_1 = (
                np.array([])
                if np.all(ic) or not self.cfg.student_loss_type == 'topr'
                else np.array(student_ratio_clipped_0_1_list[cc])
            )
            student_incorrect_ratio_clipped_0_1 = (
                np.array([])
                if np.all(cc) or not self.cfg.student_loss_type == 'topr'
                else np.array(student_ratio_clipped_0_1_list[ic])
            )

            pfx = f"{prefix}/" if prefix != "" else prefix
            log_dict.update(
                {
                    f"{pfx}avg_kl_ss_reward_ratio": avg_kl_ss_reward_ratio,
                    f"{pfx}avg_student_reward": avg_student_reward,
                    f"{pfx}avg_teacher_reward": avg_teacher_reward,
                    f"{pfx}avg_student_teacher_kl": avg_student_teacher_kl,
                    f"{pfx}avg_window_kl": avg_window_kl,
                    f"{pfx}avg_student_teacher_kl_mean": avg_student_teacher_kl_mean,
                    f"{pfx}avg_student_teacher_kl_max": avg_student_teacher_kl_max,
                    f"{pfx}avg_teacher_match_reward": avg_teacher_match_reward,
                    f"{pfx}avg_teacher_correct_match_reward": avg_teacher_correct_match_reward,
                    f"{pfx}avg_teacher_incorrect_match_reward": avg_teacher_incorrect_match_reward,
                    f"{pfx}avg_ss_reward_mean": avg_ss_reward_mean,
                    f"{pfx}avg_ss_reward_min": avg_ss_reward_min,
                    f"{pfx}avg_ss_reward": avg_ss_reward,
                    f"{pfx}avg_correct_kl_mean": avg_correct_kl_mean,
                    f"{pfx}avg_incorrect_kl_mean": avg_incorrect_kl_mean,
                    f"{pfx}avg_correct_kl_sum": avg_correct_kl_sum,
                    f"{pfx}avg_incorrect_kl_sum": avg_incorrect_kl_sum,
                    f"{pfx}avg_correct_kl_max": avg_correct_kl_max,
                    f"{pfx}avg_incorrect_kl_max": avg_incorrect_kl_max,
                    f"{pfx}avg_correct_ss_reward_mean": avg_correct_ss_reward_mean,
                    f"{pfx}avg_incorrect_ss_reward_mean": avg_incorrect_ss_reward_mean,
                    f"{pfx}avg_correct_ss_reward_min": avg_correct_ss_reward_min,
                    f"{pfx}avg_incorrect_ss_reward_min": avg_incorrect_ss_reward_min,
                    f"{pfx}avg_incorrect_incorect": 0 if len(ii) == 0 else np.mean(ii).item(),
                    f"{pfx}avg_teacher_correct_alpha": 0 if len(teacher_correct_ratio_clipped_0_1) == 0 else np.mean(teacher_correct_ratio_clipped_0_1).item(),
                    f"{pfx}avg_teacher_incorrect_alpha": 0 if len(teacher_incorrect_ratio_clipped_0_1) == 0 else np.mean(teacher_incorrect_ratio_clipped_0_1).item(),
                    f"{pfx}avg_student_correct_alpha": 0 if len(student_correct_ratio_clipped_0_1) == 0 else np.mean(student_correct_ratio_clipped_0_1).item(),
                    f"{pfx}avg_student_incorrect_alpha": 0 if len(student_incorrect_ratio_clipped_0_1) == 0 else np.mean(student_incorrect_ratio_clipped_0_1).item(),
                    # New student incorrect alpha diagnostics
                    f"{pfx}max_student_incorrect_alpha": 0 if len(student_incorrect_ratio_clipped_0_1) == 0 else np.max(student_incorrect_ratio_clipped_0_1).item(),
                    f"{pfx}ratio_student_incorrect_alpha_gt_0.1": 0
                    if len(student_incorrect_ratio_clipped_0_1) == 0
                    else float(np.mean(student_incorrect_ratio_clipped_0_1 > 0.1)),
                }
            )

            logger.info(f"{pfx} avg_teacher_reward: {avg_teacher_reward} avg_student_teacher_kl: {avg_student_teacher_kl} avg_student_teacher_kl_max: {avg_student_teacher_kl_max} avg_ss_reward_mean {avg_ss_reward_mean} avg_ss_reward_min {avg_ss_reward_min} avg_teacher_match_reward {avg_teacher_match_reward}")
            logger.info(f"{pfx} avg_correct_ss_reward_mean: {avg_correct_ss_reward_mean} avg_incorrect_ss_reward_mean: {avg_incorrect_ss_reward_mean}")
            logger.info(f"{pfx} avg_correct_kl_mean: {avg_correct_kl_mean} avg_incorrect_kl_mean: {avg_incorrect_kl_mean} avg_correct_kl_max: {avg_correct_kl_max} avg_incorrect_kl_max: {avg_incorrect_kl_max}")

        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)

    async def _calculate_teacher_rewards(
        self,
        *,
        student_experiences,
        teacher_experiences,
        answer_indices,
        initial_teacher_scores,
        teacher_custom_rewards,
        teacher_generated,
        all_teacher_prompts,
        all_student_prompts,
        initial_scores,
        ):
        """
        Calculate teacher reward, replace student/teacher log probs with the correct ones,
        and compute TOPR ratios. Returns per-sample lists and pass@N dicts for logging/normalization.
        """
        async with Timer(
            "Calculate teacher reward, replace student or teacher log probs w/ the correct one, compute torp ratio"
        ):
            final_reward_list = []
            kl_max_list = []
            kl_mean_list = []
            kl_sum_list = []
            kl_reward_list = []
            window_kl_reward_list = []
            teacher_match_reward_list = []
            ss_reward_mean_list = []
            ss_reward_min_list = []
            ss_reward_list = []
            teacher_ratio_clipped_0_1_list = []
            student_ratio_clipped_0_1_list = []

            teacher_prompt_idx = 0
            teacher_pass_at_n_dict = defaultdict(list)
            pass_at_n_dict = defaultdict(list)
            for student_exp, teacher_exp in zip(student_experiences, teacher_experiences):

                kl_div_all = compute_approx_kl(
                    teacher_exp.action_log_probs,
                    student_exp.action_log_probs
                    if not self.cfg.reward_kl_toward_ref_model
                    else student_exp.base_action_log_probs,
                    action_mask=None,
                    use_kl_estimator_k3=self.cfg.use_kl_estimator_k3,
                    use_abs_kl=self.cfg.use_abs_kl,
                    reverse=self.cfg.reverse_kl,
                )

                offset = 0
                seq_offset = 0
                total_lengths = student_exp.info["total_length"].flatten()
                for i, (num_action, student_num_action) in enumerate(
                    zip(teacher_exp.num_actions[0], student_exp.num_actions[0])
                ):
                    na = int(num_action.item())
                    student_na = int(student_num_action.item())
                    assert (
                        na == student_na
                    ), f"student and teacher num_actions must be equal {na} {student_na}"
                    seq_len = int(total_lengths[i])
                    prompt_len = seq_len - na

                    # computing answer alignment reward
                    final_answer_start, final_answer_end = answer_indices[teacher_prompt_idx]
                    teacher_score = initial_teacher_scores[teacher_prompt_idx]
                    ss_tokens_offset = self.cfg.ss_tokens_offset
                    kl_token_offset = 6
                    answer_tokens_offset = 3

                    if (
                        teacher_score
                        and final_answer_start is not None
                        and final_answer_start < final_answer_end
                    ):
                        final_answer_start_offset, final_answer_end_offset = (
                            offset + final_answer_start - ss_tokens_offset,
                            offset + final_answer_end + ss_tokens_offset,
                        )
                        final_answer_start_prob_offset = (
                            offset + final_answer_start - answer_tokens_offset
                        )
                        final_answer_end_prob_offset = (
                            offset + final_answer_end + answer_tokens_offset
                        )

                        final_answer_log_propbs = student_exp.action_log_probs[
                            :, final_answer_start_offset:final_answer_end_offset
                        ].clone()

                        # Guard against empty window which can occur after applying offsets/clamping
                        if final_answer_log_propbs.numel() == 0:
                            s_final_answer_start, s_final_answer_end = (
                                seq_offset + prompt_len + final_answer_start,
                                seq_offset + prompt_len + final_answer_end,
                            )
                            vis_final_answer = self._detokenize(
                                student_exp.sequences[0][
                                    s_final_answer_start:s_final_answer_end
                                ]
                            )
                            logger.warning(
                                f"teacher_generated {teacher_generated[teacher_prompt_idx]}, vis_final_answer: {vis_final_answer}"
                            )
                            logger.warning(
                                f"final_answer_log_propbs is empty {final_answer_start} {final_answer_end} {final_answer_start_offset} {final_answer_end_offset} {na} {seq_len} {prompt_len} {final_answer_log_propbs}"
                            )
                            # Use the same fallback as the invalid-answer branch
                            ss_reward_mean, ss_reward_min = -2.7, -11.8
                        else:
                            ss_reward_mean = final_answer_log_propbs.mean().item()
                            ss_reward_min = final_answer_log_propbs.min().item()
                        ss_reward = (
                            ss_reward_mean
                            + self.cfg.kl_max_coef * ss_reward_min
                        )

                        start_kl, end_kl, end_full = (
                            offset,
                            offset + final_answer_start - kl_token_offset,
                            offset + na,
                        )
                    else:
                        ss_reward_mean, ss_reward_min = -2.7, -11.8
                        ss_reward = (
                            ss_reward_mean
                            + self.cfg.kl_max_coef * ss_reward_min
                        )
                        start_kl, end_kl, end_full = offset, offset + na, offset + na

                    if teacher_generated[teacher_prompt_idx] == 1:
                        if start_kl < end_full:
                            student_ratio_clipped_0_1_scalar = torch.exp(
                                self.cfg.topr_temperature
                                * (student_exp.action_log_probs[:, start_kl:end_full].sum(-1) - teacher_exp.action_log_probs[:, start_kl:end_full].sum(-1)).clamp(max=0.0)
                            )
                        else:
                            student_ratio_clipped_0_1_scalar = torch.tensor(0)
                    else:
                        student_ratio_clipped_0_1_scalar = torch.tensor(1)

                    ss_reward_mean_list.append(ss_reward_mean)
                    ss_reward_min_list.append(ss_reward_min)
                    ss_reward_list.append(ss_reward)

                    if teacher_generated[teacher_prompt_idx] != 1:
                        kl_reward = kl_mean = kl_sum = kl_max = window_kl_reward = torch.tensor(
                            0.0, device=kl_div_all.device
                        )
                    else:
                        # Compute KL only over the explanation tokens before the final answer.
                        # Guard against cases where the window is empty (e.g., very short explanations),
                        # which would make reductions over an empty dimension invalid.
                        if end_kl <= start_kl:
                            kl_max = torch.tensor(0.0, device=kl_div_all.device)
                            kl_mean = torch.tensor(0.0, device=kl_div_all.device)
                            kl_sum = torch.tensor(0.0, device=kl_div_all.device)
                            kl_reward = torch.tensor(0.0, device=kl_div_all.device)
                            window_kl_reward = torch.tensor(0.0, device=kl_div_all.device)
                        else:
                            kl_episode = kl_div_all[:, start_kl:end_kl].clone()
                            kl_max = torch.max(kl_episode.abs(), dim=-1)[0]
                            kl_mean = masked_mean(kl_episode, None, dim=-1)
                            kl_sum = kl_episode.sum(dim=-1)
                            if self.cfg.reward_kl_reduction == "mean":
                                kl_reward = -kl_mean- self.cfg.kl_max_coef * kl_max
                            elif self.cfg.reward_kl_reduction == "sum":
                                kl_reward = -kl_sum - self.cfg.kl_max_coef * kl_max

                            kl_reward = torch.clamp(kl_reward, min=-self.cfg.kl_reward_clamp)

                            # Optional: rolling-window KL loss (max over mean of windows of size N)
                            window_kl_reward = torch.tensor(0.0, device=kl_div_all.device)
                            if self.cfg.kl_loss_window_size > 0 and  self.cfg.kl_window_loss_coef != 0:
                                # Compute window means along the sequence axis
                                # kl_episode shape: [B(=1), L]
                                if kl_episode.size(-1) >= self.cfg.kl_loss_window_size:
                                    pooled = F.avg_pool1d(
                                        kl_episode.unsqueeze(1), kernel_size=self.cfg.kl_loss_window_size, stride=1
                                    ).squeeze(1)  # [1, L-win_sz+1]
                                    max_mean = torch.max(pooled, dim=-1)[0]  # [1]
                                    window_kl_reward = -max_mean
                                    window_kl_reward = torch.clamp(
                                        window_kl_reward, min=-self.cfg.kl_reward_clamp
                                    )

                    window_kl_reward_list.append(window_kl_reward.item())

                    match_reward_check = teacher_custom_rewards[teacher_prompt_idx][-1]
                    match_reward = teacher_exp.info["custom_rewards"][i][-1]
                    assert (
                        match_reward_check == match_reward
                    ), "match_reward_check and match_reward must be equal"
                    # if teacher_score:
                    final_teacher_reward = (
                        self.cfg.topr_reward_coef * student_ratio_clipped_0_1_scalar
                        + self.cfg.ss_reward_coef * ss_reward_list[-1]
                        + self.cfg.reward_kl_coef * kl_reward
                        + self.cfg.kl_window_loss_coef * window_kl_reward
                        + self.cfg.reward_match_coef * match_reward
                    )
                    final_reward_list.append(final_teacher_reward.item())
                    # else:
                    #     final_reward_list.append(self.cfg.teacher_incorrect_reward)
                    teacher_pass_at_n_dict[all_teacher_prompts[teacher_prompt_idx]].append(final_reward_list[-1])
                    # For student normalization, optionally use signed exp(ss_reward)
                    if self.cfg.use_ss_reward_for_student:
                        signed = (
                            1.0 if initial_scores[teacher_prompt_idx] == 1 else -1.0
                        )
                        student_norm_score = float(np.exp(ss_reward_list[-1]) * signed)
                    else:
                        student_norm_score = float(initial_scores[teacher_prompt_idx])
                    pass_at_n_dict[all_student_prompts[teacher_prompt_idx]].append(student_norm_score)

                    kl_reward_list.append(kl_reward.item())
                    kl_max_list.append(kl_max.item())
                    kl_mean_list.append(kl_mean.item())
                    kl_sum_list.append(kl_sum.item())
                    teacher_match_reward_list.append(match_reward.item())

                    student_exp.info["loss_type"] = torch.tensor(compute_loss_type_hash(self.cfg.student_loss_type)).unsqueeze(0).float()

                    teacher_exp.info["loss_type"] = torch.tensor(compute_loss_type_hash(self.cfg.teacher_loss_type)).unsqueeze(0).float()

                    # compute ratio_clipped_0_1 for TOPR
                    if self.cfg.student_loss_type == "topr":
                        if teacher_generated[teacher_prompt_idx] == 1 and start_kl < end_full:
                            if self.cfg.topr_type == 0:
                                student_ratio_clipped_0_1_scalar = torch.exp(
                                    self.cfg.topr_temperature
                                    * (
                                        student_exp.action_log_probs[
                                            :, start_kl:end_full
                                        ].sum(-1)
                                        - teacher_exp.action_log_probs[
                                            :, start_kl:end_full
                                        ].sum(-1)
                                    ).clamp(max=0.0)
                                )
                                student_exp.ratio_clipped_0_1[
                                    :, start_kl:end_full
                                ] = student_ratio_clipped_0_1_scalar
                            elif self.cfg.topr_type == 1:
                                student_ratio_clipped_0_1_scalar = torch.exp(
                                    self.cfg.topr_temperature
                                    * (
                                        student_exp.base_action_log_probs[
                                            :, start_kl:end_full
                                        ]
                                        - teacher_exp.action_log_probs[
                                            :, start_kl:end_full
                                        ]
                                    ).clamp(max=0.0)
                                )
                                student_exp.ratio_clipped_0_1[
                                    :, start_kl:end_full
                                ] = student_ratio_clipped_0_1_scalar
                                student_ratio_clipped_0_1_scalar = (
                                    student_ratio_clipped_0_1_scalar.mean()
                                )
                            elif self.cfg.topr_type == 2:
                                diff = (
                                    student_exp.action_log_probs[:, start_kl:end_full]
                                    - teacher_exp.action_log_probs[:, start_kl:end_full]
                                )
                                suffix = torch.flip(torch.flip(diff, dims=[-1]).cumsum(dim=-1), dims=[-1])
                                student_ratio_vec = torch.exp(
                                    self.cfg.topr_temperature * suffix.clamp(max=0.0)
                                )
                                student_exp.ratio_clipped_0_1[:, start_kl:end_full] = student_ratio_vec
                                student_ratio_clipped_0_1_scalar = student_ratio_vec.mean()
                                # logger.info(f'2 student_ratio_clipped_0_1_scalar {student_ratio_clipped_0_1_scalar}')
                        else:
                            student_ratio_clipped_0_1_scalar = torch.tensor(1)
                        student_ratio_clipped_0_1_list.append(
                            student_ratio_clipped_0_1_scalar.item()
                        )

                    if self.cfg.teacher_loss_type == "topr":
                        if teacher_generated[teacher_prompt_idx] == 1:
                            teacher_ratio_clipped_0_1_scalar = torch.tensor(1)
                        else:
                            if start_kl < end_full:
                                if self.cfg.topr_type == 0:
                                    teacher_ratio_clipped_0_1_scalar = torch.exp(
                                        self.cfg.topr_temperature
                                        * (
                                            teacher_exp.action_log_probs[
                                                :, start_kl:end_full
                                            ].sum(-1)
                                            - student_exp.action_log_probs[
                                                :, start_kl:end_full
                                            ].sum(-1)
                                        ).clamp(max=0.0)
                                    )
                                    teacher_exp.ratio_clipped_0_1[
                                        :, start_kl:end_full
                                    ] = teacher_ratio_clipped_0_1_scalar
                                elif self.cfg.topr_type == 1:
                                    teacher_ratio_clipped_0_1_scalar = torch.exp(
                                        self.cfg.topr_temperature
                                        * (
                                            teacher_exp.action_log_probs[
                                                :, start_kl:end_full
                                            ]
                                            - student_exp.action_log_probs[
                                                :, start_kl:end_full
                                            ]
                                        ).clamp(max=0.0)
                                    )
                                    teacher_exp.ratio_clipped_0_1[
                                        :, start_kl:end_full
                                    ] = teacher_ratio_clipped_0_1_scalar
                                    teacher_ratio_clipped_0_1_scalar = (
                                        teacher_ratio_clipped_0_1_scalar.mean()
                                    )
                                elif self.cfg.topr_type == 2:
                                    diff = (
                                        teacher_exp.action_log_probs[:, start_kl:end_full]
                                        - student_exp.action_log_probs[:, start_kl:end_full]
                                    )
                                    suffix = torch.flip(torch.flip(diff, dims=[-1]).cumsum(dim=-1), dims=[-1])
                                    teacher_ratio_vec = torch.exp(
                                        self.cfg.topr_temperature * suffix.clamp(max=0.0)
                                    )
                                    teacher_exp.ratio_clipped_0_1[:, start_kl:end_full] = teacher_ratio_vec
                                    teacher_ratio_clipped_0_1_scalar = teacher_ratio_vec.mean()

                        teacher_ratio_clipped_0_1_list.append(
                            teacher_ratio_clipped_0_1_scalar.item()
                        )

                    if (
                        self.cfg.replace_all_teacher_base_logprops_w_student
                        and start_kl < end_full
                    ):
                        teacher_exp.base_action_log_probs[
                            :, start_kl:end_full
                        ] = student_exp.base_action_log_probs[
                            :, start_kl:end_full
                        ].clone()

                    if teacher_generated[teacher_prompt_idx] != 1:
                        if self.cfg.replace_teacher_logprops_w_student and start_kl < end_full:
                            teacher_exp.action_log_probs[
                                :, start_kl:end_full
                            ] = student_exp.action_log_probs[
                                :, start_kl:end_full
                            ].clone()
                        if (
                            self.cfg.replace_teacher_base_logprops_w_student
                            and start_kl < end_full
                        ):
                            teacher_exp.base_action_log_probs[
                                :, start_kl:end_full
                            ] = student_exp.base_action_log_probs[
                                :, start_kl:end_full
                            ].clone()
                    else:
                        if self.cfg.replace_student_logprops_w_teacher and start_kl < end_full:
                            student_exp.action_log_probs[
                                :, start_kl:end_full
                            ] = teacher_exp.action_log_probs[
                                :, start_kl:end_full
                            ].clone()
                        if (
                            self.cfg.replace_student_base_logprops_w_teacher
                            and start_kl < end_full
                        ):
                            student_exp.base_action_log_probs[
                                :, start_kl:end_full
                            ] = teacher_exp.base_action_log_probs[
                                :, start_kl:end_full
                            ].clone()

                    if (
                        teacher_score
                        and final_answer_start is not None
                        and final_answer_start < final_answer_end
                    ):
                        if self.cfg.teacher_explain_only:
                            if teacher_generated[teacher_prompt_idx] == 1:
                                # we programatically add the final answer, so the log prob should be 0
                                student_exp.action_log_probs[
                                    :, final_answer_start_prob_offset:final_answer_end_prob_offset
                                ] = 0
                            # mask out teacher answer for teacher if teacher_explain_only
                            teacher_exp.action_mask[
                                :, final_answer_start_prob_offset:final_answer_end_prob_offset
                            ] = 0

                    offset += na
                    teacher_prompt_idx += 1
                    seq_offset += seq_len

                    if not self.cfg.use_grpo:
                        teacher_exp.info["custom_rewards"][i][-1] = final_reward_list[-1]

                assert (
                    kl_div_all.shape[1] == offset
                ), "number of action should be the same in kl and num_actions"
                assert (
                    len(student_exp.sequences[0]) == seq_offset
                ), "student_exp.sequences must be equal to seq_offset at the end"

            assert (
                len(final_reward_list) == teacher_prompt_idx == len(all_teacher_prompts)
            ), "kl_reward_list and last teacher prompt idx and all_teacher_prompts must be equal to all teacher prompts length"

            return (
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
            )

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

    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=None,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            None,
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
        self.writer.add_scalar("Avg reward", avg_rewards, self.global_step)
        self.writer.add_scalar("Avg KL", avg_kl, self.global_step)
        self.writer.flush()
        return experience

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
        if self.cfg.use_ref_model:
            base_action_log_probs_ref = micro_infer_model(
                num_ref_dp_groups, "ref_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
            )
            base_log_probs = None

        # handle colocate critic and reward model
        if self.cfg.colocate_critic_reward and not self.cfg.colocate_all and self.critic_model is not None:
            values = await value_ref
            await self.critic_model.async_run_method("empty_cache")

        # handle colocate actor and ref model
        if (self.cfg.colocate_actor_ref or self.cfg.colocate_all) and self.cfg.use_ref_model:
            base_log_probs = await base_action_log_probs_ref
            await self.ref_model.async_run_method("empty_cache")

        # calculate rewards
        reward_refs = []
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

        if not self.cfg.use_ref_model:
            base_log_probs = [logprobs.clone() for logprobs in action_log_probs]

        r = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else None
        if not self.cfg.colocate_all:
            empty_cache_tasks = [
                self.policy_model.async_run_method("empty_cache") if not use_teacher_model else self.teacher_model.async_run_method("empty_cache"),
            ]
            if self.cfg.use_ref_model:
                empty_cache_tasks.append(self.ref_model.async_run_method("empty_cache"))
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
                    torch.ones_like(action_log_probs[i]) if self.cfg.teacher_explain_only else None,
                    response_length,
                    torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),
                    info,
                    kl,
                    torch.ones_like(action_log_probs[i]) if self.cfg.student_loss_type == 'topr' or self.cfg.teacher_loss_type == 'topr' else None,
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
        # Build dataloader, optionally using a balanced yes/no batch sampler
        if self.cfg.balance_yes_no_batches:
            batch_sampler = BalancedYesNoBatchSampler(
                dataset,
                batch_size=self.cfg.rollout_batch_size,
                drop_last=False,
                seed=getattr(self.cfg, "seed", 42),
            )
            prompts_dataloader = DataLoader(
                dataset, batch_sampler=batch_sampler, collate_fn=dataset.collate_fn, num_workers=8
            )
        else:
            prompts_dataloader = DataLoader(
                dataset, batch_size=self.cfg.rollout_batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8
            )
        self.num_update_steps_per_episodes = (
            len(dataset) * self.cfg.n_samples_per_prompt // self.cfg.train_batch_size * self.cfg.max_epochs
        )
        max_steps = math.ceil(self.cfg.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps
        # Expose total training steps to strategy args so schedulers (e.g., linear) can decay to zero
        logger.info(f"len(dataset) // self.cfg.train_batch_size: {len(dataset) // self.cfg.rollout_batch_size}")
        self.total_num_training_steps = len(dataset) // self.cfg.rollout_batch_size * self.cfg.num_episodes
        logger.info(f"Total number of training steps: {self.total_num_training_steps}")
        setattr(self.strategy.args, "total_num_training_steps", self.total_num_training_steps)

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
                num_gpus_per_actor=0.2 if not self.cfg.separate_teacher_model else 0.1,
            )
            # Create separate teacher model if flag is enabled
            if cfg.separate_teacher_model:
                teacher_model = PPORayActorGroup(
                    cfg.actor_num_nodes,
                    cfg.actor_num_gpus_per_node,
                    PolicyRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.1,
                )
            if self.cfg.use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg.ref_num_nodes,
                    cfg.ref_num_gpus_per_node,
                    RefRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2 if not self.cfg.separate_teacher_model else 0.1,
                )
            else:
                ref_model = None

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
                if cfg.separate_teacher_model:
                    if cfg.critic_pretrain:
                        num_gpus_per_actors = [0.25]
                    else:
                        num_gpus_per_actors = [0.4, 0.2]
                else:
                    if cfg.critic_pretrain:
                        num_gpus_per_actors = [0.3]
                    else:
                        num_gpus_per_actors = [0.7, 0.25]

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=num_gpus_per_actors[0] if pg else 1,
            )
            if self.cfg.separate_teacher_model:
                teacher_model = PPORayActorGroup(
                    cfg.actor_num_nodes,
                    cfg.actor_num_gpus_per_node,
                    PolicyRayActor,
                    pg=pg,
                    num_gpus_per_actor=num_gpus_per_actors[0] if pg else 1,
                )
            if cfg.use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg.ref_num_nodes,
                    cfg.ref_num_gpus_per_node,
                    RefRayActor,
                    pg=pg,
                    num_gpus_per_actor=num_gpus_per_actors[-1] if pg else 1,
                )
            else:
                ref_model = None

            # if colocated, create placement group for critic and reward model explicitly.
            if cfg.critic_pretrain:
                if self.cfg.colocate_critic_policy:
                    critic_model = PPORayActorGroup(
                        cfg.critic_num_nodes,
                        cfg.critic_num_gpus_per_node,
                        CriticRayActor,
                        pg=pg,
                        num_gpus_per_actor=num_gpus_per_actors[0] if pg else 1,
                    )
                    reward_models = None
                else:
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
            else:
                reward_models = None
                critic_model = None

        if not cfg.colocate_all:
            refs = []
            if ref_model is not None:
                refs.extend(ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            logger.info(f"init policy from {cfg.pretrain}")
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            if cfg.separate_teacher_model:
                logger.info(f"init teacher from {cfg.teacher_pretrain}")
                refs.extend(teacher_model.async_init_model_from_pretrained(self.strategy, cfg.teacher_pretrain))
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
            if ref_model is not None:
                await asyncio.gather(*ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
            await policy_model.offload_to_cpu()
            if cfg.separate_teacher_model:
                await asyncio.gather(*teacher_model.async_init_model_from_pretrained(self.strategy, cfg.teacher_pretrain))
                await teacher_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
                await teacher_model.offload_to_cpu()
            if cfg.critic_pretrain:
                await asyncio.gather(*critic_model.async_init_model_from_pretrained(self.strategy, cfg.critic_pretrain))
                await critic_model.offload_to_cpu()
            if cfg.reward_pretrain:
                for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
                    await asyncio.gather(*reward_model.async_init_model_from_pretrained(self.strategy, reward_pretrain))

        self.policy_model = policy_model
        if cfg.separate_teacher_model:
            self.teacher_model = teacher_model
        self.critic_model = critic_model
        self.ref_model = ref_model
        self.reward_model = reward_models

        logger.info("init policy/teacher/ref/critic/reward models done")

    async def ppo_local_train_policy(self, model, replay_buffers: List[NaiveReplayBuffer], global_steps: int, prefix: str = "", backlog: bool = False):
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer(f"{prefix.capitalize()} policy model training"):
                status = await model.async_ppo_train(global_steps, replay_buffers)
            # Log with prefix for separate tracking
            metric_prefix = f"{prefix}_" if prefix else ""
            self.writer.add_scalar(f"{metric_prefix}ppo_clip_count", status[0]["clip_ratio"], global_steps)
            self.writer.add_scalar(f"{metric_prefix}policy_update_steps", status[0]["policy_update_steps"], global_steps)
            self.writer.add_scalar(f"{metric_prefix}policy_entropy", status[0]["entropy"], global_steps)
            # Log learning rate for policy (student unprefixed, teacher with 'teacher_')
            if "actor_lr" in status[0]:
                self.writer.add_scalar(f"{metric_prefix}actor_lr", status[0]["actor_lr"], global_steps)
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
            # Log learning rate for critic (student unprefixed, teacher with 'teacher_')
            if "critic_lr" in status[0]:
                self.writer.add_scalar(f"{metric_prefix}critic_lr", status[0]["critic_lr"], global_steps)
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
        # ToDo: hardcoded action mask = None, need to fix it later
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=None, #experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            None, #experience.action_mask,
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
            responses_logprobs = []
            for i, prompt in enumerate(prompts):
                content = outputs[i].outputs[0].text
                finish_reasons.append(outputs[i].outputs[0].finish_reason)
                responses.append(content)
                if outputs[i].prompt_logprobs:
                    prompt_logprobs.append(outputs[i].prompt_logprobs)
                if outputs[i].outputs[0].logprobs:
                    responses_logprobs.append(outputs[i].outputs[0].logprobs)
            if len(responses_logprobs) > 0:
                return (
                    responses,
                    finish_reasons,
                    responses_logprobs,
                )
            if len(prompt_logprobs) > 0:
                return (
                    responses,
                    finish_reasons,
                    prompt_logprobs,
                )
            else:
                return responses, finish_reasons

        return generate

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

    async def _offload_vllm_engines(self, engines=None):
        engines = self.vllm_engines if engines is None else engines
        if not engines:
            return
        offload_tasks = []
        for engine in engines:
            offload_tasks.append(engine.offload_to_cpu.remote())
        await asyncio.gather(*offload_tasks)

    async def _backload_vllm_engines(self, engines=None):
        engines = self.vllm_engines if engines is None else engines
        if not engines:
            return
        backload_tasks = []
        for engine in engines:
            backload_tasks.append(engine.backload_to_gpu.remote())
        await asyncio.gather(*backload_tasks)

    async def _sync_policy_weights_to_vllm(self):
        if self._dual_vllm_enabled and self._vllm_current_role != "student":
            raise RuntimeError("Attempting to sync policy weights while vLLM engines are not in student mode")
        if self.cfg.colocate_all:
            await self.policy_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.policy_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)

    async def _sync_teacher_weights_to_vllm(self):
        if self._dual_vllm_enabled and self._vllm_current_role != "teacher":
            raise RuntimeError("Attempting to sync teacher weights while vLLM engines are not in teacher mode")
        if self.cfg.colocate_all:
            await self.teacher_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.teacher_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)

    async def _major_sync_policy_weights_to_vllm(self):
        await self._ensure_vllm_role("student")
        if self.cfg.colocate_all:
            # await self._backload_vllm_engines(self.vllm_engines)
            await self.policy_model.backload_to_gpu()
            await self._sync_policy_weights_to_vllm()
            await self.policy_model.offload_to_cpu()
        else:
            await self._sync_policy_weights_to_vllm()

    async def _major_sync_teacher_weights_to_vllm(self):
        if not self.cfg.separate_teacher_model:
            return
        await self._ensure_vllm_role("teacher")
        if self.cfg.colocate_all:
            # await self._backload_vllm_engines(self.vllm_engines)
            await self.teacher_model.backload_to_gpu()
            await self._sync_teacher_weights_to_vllm()
            await self.teacher_model.offload_to_cpu()
        else:
            await self._sync_teacher_weights_to_vllm()

    async def _destroy_vllm_engines(self):
        """Terminate current vLLM Ray actors to free GPU memory."""
        for eng in self.vllm_engines:
            ray.kill(eng)
        self.vllm_engines = []

        if not self.cfg.colocate_all:
            for pg in self._vllm_pg_handles:
                ray.util.remove_placement_group(pg)

    async def _recreate_vllm_engines(self, *, pretrain: str, role: str = "student"):
        """Create fresh vLLM engines with the provided model, re-init comm groups, and return handles.

        This is used when teacher and student models have different architectures
        and we want to switch the vLLM backend between them infrequently.
        """
        # 1) Tear down old engines
        logger.info(f"Recreating vLLM engines for role '{role}' with pretrain '{pretrain}'")
        logger.info("Destroying old vLLM engines...")
        await self._destroy_vllm_engines()

        # 2) Create new engines using the same resource config
        from orz.ppo.utils import create_vllm_engines
        logger.info("Creating new vLLM engines...")
        self.vllm_engines = create_vllm_engines(
            self.cfg.vllm_num_engines,
            self.cfg.vllm_tensor_parallel_size,
            pretrain,
            self.cfg.seed,
            self.cfg.enable_prefix_caching,
            self.cfg.enforce_eager,
            self.cfg.max_len,
            self.cfg.colocate_all,
            self.cfg.enable_chunked_prefill,
            self.cfg.max_num_batched_tokens,
            self.cfg.gpu_memory_utilization,
            self.cfg.micro_rollout_batch_size,
            self.colocate_pg,
            # return_pg_handles=True,
        )

        # 3) Re-initialize the comm groups on policy/teacher models for these engines
        if role == "student":
            await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)
        elif role == "teacher":
            await self.teacher_model.async_run_method("_init_teacher_vllm_engines_actor_group", self.vllm_engines)

    async def _ensure_colocated_vllm_role(self, role: str):
        logger.info(f"Ensuring colocated vLLM role '{role}' {self._vllm_current_role}")
        if not self._dual_vllm_enabled:
            return False
        if role == self._vllm_current_role:
            return True
        logger.info(f"Switching colocated vLLM engines to role '{role}'")
        target_engines = self.student_vllm_engines if role == "student" else self.teacher_vllm_engines
        if target_engines is None:
            raise RuntimeError(f"Requested vLLM role '{role}' but corresponding engines are not initialized")
        await self._offload_vllm_engines(self.vllm_engines)
        await self._backload_vllm_engines(target_engines)
        self.vllm_engines = target_engines
        self._vllm_current_role = role
        return True

    async def _ensure_vllm_role(self, role: str):
        """Ensure vLLM engines are created for the requested role ('student'|'teacher').

        When cfg.vllm_recreate_on_switch is False, this is a no-op.
        When True and role changes, we recreate engines with the appropriate model
        and sync corresponding weights to the engines.
        """
        if await self._ensure_colocated_vllm_role(role):
            return
        if not self.cfg.vllm_recreate_on_switch:
            return
        if role == self._vllm_current_role:
            return

        # Choose model for the role
        pretrain_model = self.cfg.teacher_pretrain if role == "teacher" else self.cfg.pretrain

        # Recreate and re-sync
        await self._recreate_vllm_engines(pretrain=pretrain_model, role=role)
        logger.info(f"recreate_vllm_engines for role '{role}' done")
        await self._backload_vllm_engines()
        logger.info(f"_backload_vllm_engines for role '{role}' done")

        self._vllm_current_role = role

    async def _sync_policy_weights_to_teacher(self):
        async with Timer("Saving current policy"):
            await self.policy_model.async_save_model(self.tokenizer, '_current')
        model_dir = os.path.join(self.cfg.save_path, f"iter_current", "policy", 'model.safetensors')
        if self.cfg.colocate_all:
            await self.teacher_model.backload_to_gpu()
        async with Timer("Loading policy weights to teacher model"):
            await self.teacher_model.async_run_method("_load_policy_from_dir", model_dir)
            # Reset optimizer/scheduler state on teacher to avoid stale momentum
            await self.teacher_model.async_run_method("_reset_optimizer_state", True)
            await self.teacher_model.offload_to_cpu()
            await self.teacher_model.backload_to_gpu()
        if self.cfg.colocate_all:
            await self.teacher_model.offload_to_cpu()


def compute_loss_type_hash(loss_type):
    if loss_type == 'ppo':
        loss_type_hash = 1
    elif loss_type == 'topr':
        loss_type_hash = 2
    elif loss_type == 'sft':
        loss_type_hash = 3
    else:
        assert False, f"student loss type {loss_type} must be ppo, sft or topr"
    return loss_type_hash
