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
    create_student_prompt,
    create_teacher_prompt_from_answer,
    extract_visible_reasoning,
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
        vllm_pg_handles: Optional[list] = [],
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.vllm_engines = vllm_engines
        self.prompts_dataloader = self.build_dataloader(train_dataset)
        self.colocate_pg = colocate_pg
        # Track which role the current vLLM engines are set up for.
        # Engines created at startup use cfg.pretrain (student) by default.
        self._vllm_current_role: Optional[str] = "student" if cfg.vllm_recreate_on_switch else None
        self._vllm_pg_handles = vllm_pg_handles

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

    def log_student_generation_examples(
        self,
        *,
        all_student_prompts: List[str],
        all_teacher_prompts: List[str],
        outputs: List[Any],
        final_answers: List[Any],
        initial_scores: List[Any],
        initial_teacher_scores: List[Any],
        indices_incorrect: List[int],
        step: Optional[int] = None,
    ) -> None:
        n = min(5, len(all_student_prompts))
        table_data: List[List[Any]] = []
        for i in range(n):
            table_data.append([
                all_student_prompts[i],
                all_teacher_prompts[i],
                outputs[i],
                final_answers[i],
                bool(initial_scores[i]),
                bool(initial_teacher_scores[i]),
            ])
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
            name="student_generation_examples",
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

    def log_student_responses_by_prompt(
        self,
        *,
        student_responses_by_prompt: Dict[str, List[str]],
        student_final_answers_by_prompt: Optional[Dict[str, List[str]]] = None,
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
            finals = []
            if student_final_answers_by_prompt is not None:
                finals = student_final_answers_by_prompt.get(p, [])

            for idx, resp in enumerate(responses):
                final = finals[idx] if idx < len(finals) else ""
                rows.append([p, idx, resp, final])

        self._log_wandb_table(
            name="student_responses_by_prompt",
            columns=[
                "prompt",
                "response_idx",
                "student_response",
                "student_final_answer",
            ],
            data=rows,
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

    def log_adversarial_examples(
        self,
        all_opponents_prompts: List[str],
        opponents_custom_rewards: List[str],
        adv_prompts: List[str],
        adv_outputs: List[Any],
        adv_final_answers: List[Any],
        adv_extras: List[dict],
        adv_initial_scores: List[Any],
        index_group_dict,
        step: Optional[int] = None,
    ) -> None:
        n = min(8, len(adv_prompts))
        indices = [-i-1 for i in range(n)] + [i for i in range(n)]
        table_data: List[List[Any]] = []
        for idx in indices:
            index_group = index_group_dict[adv_prompts[idx]]
            if len(index_group) == 1:
                prompt_index = index_group
                prompt_index_2 = None
            else:
                prompt_index, prompt_index_2 = index_group

            table_data.append([
                student_prompts[prompt_index],
                teacher_prompts[prompt_index],
                teacher_prompts[prompt_index_2] if prompt_index_2 is not None else "",
                adv_prompts[idx],
                adv_outputs[idx],
                adv_final_answers[idx],
                adv_extras[idx].get("teacher_answer", ""),
                bool(adv_initial_scores[idx]),
                bool(adv_initial_teacher_scores[idx]),
                combined_teacher_custom_rewards[prompt_index][-1].item(),
                combined_teacher_custom_rewards[prompt_index_2][-1].item() if prompt_index_2 is not None else None,
                combined_custom_rewards[prompt_index][-1].item(),
                combined_custom_rewards[prompt_index_2][-1].item() if prompt_index_2 is not None else None,

            ])
        self._log_wandb_table(
            name="adversarial_examples",
            columns=[
                "student_prompts",
                "teacher_prompts_1",
                "teacher_prompts_2",
                "adv_prompt",
                "adv_response",
                "adv_final_answer",
                "teacher_answer",
                "student_correct",
                "teacher_correct",
                "teacher_adv_match_reward",
                "teacher_adv_match_reward_2",
                "student_adv_match_reward",
                "student_adv_match_reward_2",
            ],
            data=table_data,
            step=step,
        )

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

    def _create_opponents_prompts(
        self,
        all_student_prompts: List[str],
        all_extras: List[dict],
        bos_token: str,
    ) -> Tuple[List[str], List[str], List[dict], List[int], List[int]]:
        """
        Create complementary teacher prompts based on configured augment_strategy.
        Returns:
          - all_teacher_prompts: teacher prompts for generation
          - aug_all_student_prompts: repeated student prompts aligned to teacher prompts
          - aug_all_extras: repeated extras aligned to teacher prompts
          - indices_incorrect: representative incorrect indices for logging
          - new_indicess: indices mapping to representative examples for logs
        """
        all_teacher_prompts: List[str] = []
        aug_all_student_prompts: List[str] = []
        aug_all_extras: List[dict] = []
        indices_incorrect: List[int] = []
        new_indicess: List[int] = []

        # Track which teacher prompts were already added so we can repeat
        # each unique prompt exactly n_samples_per_prompt times.
        added_teacher_prompt_keys = set()

        if self.cfg.augment_strategy == "distill":
            return all_student_prompts, all_student_prompts, all_extras, indices_incorrect, np.arange(8)


        allowed_strategies = {"correct", "wrong", "yes_no", "only_wrong", "only_correct", "opposite", "correct_incorrect"}
        assert self.cfg.augment_strategy in allowed_strategies, (
            f"augment_strategy must be one of {allowed_strategies}, got {self.cfg.augment_strategy}"
        )
        if self.cfg.augment_strategy in {"only_wrong", "only_correct", "opposite"}:
            assert (
                self.cfg.generate_with_student
            ), f"{self.cfg.augment_strategy} strategy require student generation to be enabled"

        # Precompute candidate negatives if needed
        candidate_student_negs = defaultdict(list)
        dataset_answer_pool: List[str] = []
        if self.cfg.augment_strategy == "correct_incorrect":
            if self.cfg.generate_with_student:
                for sp, ex, fa, sc in zip(all_student_prompts, all_extras, final_answers, initial_scores):
                    if not sc and len(fa.strip()) > 0:
                        candidate_student_negs[sp].append(fa)

            dataset_answer_pool = [ex["answer"] for ex in all_extras if len(ex["answer"]) > 0]


        for i, (extra, student_prompt) in enumerate(zip(all_extras, all_student_prompts)):
            include = True
            teacher_answer = None
            teacher_answers = None
            is_correct = None

            teacher_score = initial_teacher_scores[i] if self.cfg.generate_with_student else None
            student_score = initial_scores[i] if self.cfg.generate_with_student else None
            final_answer = final_answers[i] if self.cfg.generate_with_student else None

            if self.cfg.augment_strategy == "correct":
                representative_incorrect = bool(student_score is not None and not student_score)
                teacher_answer = extra["answer"]
                is_correct = True

            elif self.cfg.augment_strategy == "wrong":
                representative_incorrect = bool(student_score is not None and not student_score)
                if student_score and teacher_yes[i]:
                    teacher_answer = self.no_token()
                elif student_score and teacher_no[i]:
                    teacher_answer = self.yes_token()
                else:
                    continue
                is_correct = False

            elif self.cfg.augment_strategy == "correct_incorrect":
                representative_incorrect = not bool(student_score)
                correct_ans = extra["answer"]
                neg_ans = None
                if self.cfg.generate_with_student:
                    neg_cands = candidate_student_negs.get(student_prompt, [])
                    if len(neg_cands) > 0:
                        neg_ans = neg_cands[-1]
                if neg_ans is None:
                    # Fallback: sample a different dataset answer
                    if self.train_teacher:
                        neg_ans = dataset_answer_pool[-1]
                    else:
                        neg_ans = random.choice(dataset_answer_pool)
                assert neg_ans is not None, "Negative answer must be not None by now"
                teacher_answers = [correct_ans, neg_ans]
                is_corrects = [True, False]

            elif self.cfg.augment_strategy == "yes_no":
                representative_incorrect = bool(student_score is not None and not student_score)
                teacher_answers = [self.yes_token(), self.no_token()]
                assert extra["answer"] in ["yes", "no"], (
                    f"Ground-truth answer must be yes or no for yes_no strategy, got {extra['answer']}"
                )
                is_corrects = [extra["answer"] == "yes", extra["answer"] == "no"]

            elif self.cfg.augment_strategy == "only_wrong":
                if teacher_score and (not student_score):
                    representative_incorrect = True
                    if teacher_yes[i]:
                        teacher_answer = self.no_token()
                    elif teacher_no[i]:
                        teacher_answer = self.yes_token()
                    else:
                        assert False, f"final_answer {final_answer} must be yes or no"
                else:
                    include = False
                    representative_incorrect = False
                is_correct = True

            elif self.cfg.augment_strategy == "only_correct":
                if teacher_score and student_score:
                    representative_incorrect = False
                    if teacher_yes[i]:
                        teacher_answer = self.no_token()
                    elif teacher_no[i]:
                        teacher_answer = self.yes_token()
                    else:
                        assert False, f"final_answer {final_answer} must be yes or no"
                else:
                    include = False
                    representative_incorrect = True
                is_correct = False

            elif self.cfg.augment_strategy == "opposite":
                if teacher_score:
                    representative_incorrect = not bool(student_score)
                    if teacher_yes[i]:
                        teacher_answer = self.no_token()
                    elif teacher_no[i]:
                        teacher_answer = self.yes_token()
                    else:
                        assert False, f"final_answer {final_answer} must be yes or no"
                    is_correct = not bool(student_score)
                else:
                    include = False
                    representative_incorrect = False
            else:
                assert False, "One student augmenting strategy must be chosen"

            if not include:
                continue

            teacher_answers = [teacher_answer] if teacher_answers is None else teacher_answers
            is_corrects = [is_correct] if len(teacher_answers) == 1 else is_corrects

            repeat_prompts = not (self.cfg.use_student_history and self.train_student)

            random_ans_idx = np.random.randint(0, 2)
            for ans_idx, (ans, is_corr) in enumerate(zip(teacher_answers, is_corrects)):
                teacher_prompt = create_teacher_prompt_from_answer(
                    extra["dialogue"], ans, bos_token, cfg=self.cfg, is_correct=is_corr
                )
                key = teacher_prompt
                if key in added_teacher_prompt_keys and repeat_prompts:
                    continue

                new_extra = dict(extra)
                new_extra["teacher_answer"] = ans

                index = len(all_teacher_prompts)
                if  self.cfg.verifier_use_mixed_chains and self.cfg.adversarial_training and self.cfg.augment_strategy == "yes_no" and self.cfg.repeat_randomly_once:
                    if ans_idx == random_ans_idx:
                        repeats = 1
                    else:
                        repeats = self.cfg.n_samples_per_prompt
                elif not repeat_prompts:
                    repeats = self.cfg.n_teacher_samples_per_prompt if self.cfg.n_teacher_samples_per_prompt > 0 else 1
                elif is_corr and self.cfg.teacher_k_correct_per_prompt > 0:
                    repeats = self.cfg.teacher_k_correct_per_prompt
                else:
                    repeats =  self.cfg.n_samples_per_prompt
                all_teacher_prompts.extend([teacher_prompt] * repeats)
                aug_all_student_prompts.extend([student_prompt] * repeats)
                aug_all_extras.extend([dict(new_extra)] * repeats)

                new_indicess.append(index)
                if representative_incorrect:
                    indices_incorrect.append(index)
                added_teacher_prompt_keys.add(key)

        if self.cfg.augment_strategy in ["correct", "opposite"] and len(all_extras) != len(aug_all_extras):
            logger.warning(
                f"extras don't match augmented extras in length, {len(all_extras)} {len(aug_all_extras)}"
            )
        elif self.cfg.augment_strategy in ["yes_no", "correct_incorrect"] and 2 * len(all_extras) != len(aug_all_extras):
            logger.warning(
                f"double extras don't match augmented extras in length, {2 * len(all_extras)} {len(aug_all_extras)}"
            )

        return (
            all_teacher_prompts,
            aug_all_student_prompts,
            aug_all_extras,
            indices_incorrect,
            new_indicess,
        )

    def _build_adversarial_student_prompts(
        self,
        combined_outputs: List[str],
        combined_all_teacher_prompts: List[str],
        combined_all_student_prompts: List[str],
        combined_extras: List[dict],
        teacher_generated: List[bool],
        bos_token: str,
        combined_final_answers: List[str],
    ) -> Tuple[List[str], List[str], List[dict], List[bool], List[str], List[List[int]]]:
        """
        Build adversarial continuation student prompts from teacher explanations.
        Keeps one new prompt per teacher sample (already repeated for GRPO).
        Returns adv_prompts, adv_extras.
        """

        extracted_reasonings: List[str] = []
        for resp in combined_outputs:
            extracted_reasonings.append(extract_visible_reasoning(resp, use_say=self.cfg.teacher_use_say_operator))

        adv_prompts: List[str] = []
        adv_extras: List[dict] = []
        # For each adversarial group (one mixed pair or one single),
        # record which original teacher indices should receive the verifier reward.
        adv_teacher_index_groups: List[List[int]] = []

        # Special handling for yes_no augmentation: mix YES/NO teacher chains
        if self.cfg.augment_strategy == "yes_no" and self.cfg.verifier_use_mixed_chains:
            assert not self.cfg.generate_with_student, "Cannot mix chains when student generation is enabled"
            # Group teacher generations by the underlying student prompt so we can
            # collect one YES chain and one NO chain per base dialogue.
            group: Dict[str, Dict[str, Any]] = {}

            for i, (t_prompt, s_prompt, prev_r, extra, tgenerated) in enumerate(
                zip(
                    combined_all_teacher_prompts,
                    combined_all_student_prompts,
                    extracted_reasonings,
                    combined_extras,
                    teacher_generated,
                )
            ):

                # Only consider teacher-generated samples that have explicit teacher answers
                if not tgenerated:
                    continue
                label = extra["teacher_answer"]
                if label not in ("yes", "no"):
                    assert False, f"teacher_answer must be yes or no to mix chains, got {label}"

                key = s_prompt  # group by base student prompt
                if key not in group:
                    new_extra = dict(extra)
                    # remove teacher answer to evaluate teacher_yes and teacher_no correctly
                    new_extra['teacher_answer'] = None

                    group[key] = {
                        "s_prompt": s_prompt,
                        "t_prompts": {"yes": [], "no": []},
                        "extra": new_extra,
                        "chains": {"yes": [], "no": []},
                        "t_indices": {"yes": [], "no": []},
                    }

                group[key]["chains"][label].append(prev_r)
                group[key]["t_indices"][label].append(i)
                group[key]["t_prompts"][label].append(t_prompt)

            # Build mixed previous reasoning when both sides exist; otherwise fallback to single
            for key, bundle in group.items():
                extra = bundle["extra"]
                yes_list = bundle["chains"]["yes"]
                no_list = bundle["chains"]["no"]
                yes_indices = bundle["t_indices"]["yes"]
                no_indices = bundle["t_indices"]["no"]

                if not self.cfg.repeat_randomly_once:
                    assert len(yes_list) == len(no_list) and len(yes_list) > 0, "yes and no lists must match and be non-empty"

                list_len = max(len(yes_list), len(no_list))
                for i in range(list_len):
                    i_yes = i % len(yes_list)
                    i_no = i % len(no_list)
                    # logger.info(f"Mixing adversarial reasoning chains: YES index {i_yes}, NO index {i_no}")
                    mixed_prev = f"[Answer: yes]: {yes_list[i_yes]} [Answer: no]: {no_list[i_no]}"
                    new_prompt = create_student_prompt(
                        extra["dialogue"], bos_token=bos_token, previous_reasoning=mixed_prev, cfg=self.cfg
                    )

                    for _ in range(self.cfg.adv_n_samples_per_prompt):
                        adv_prompts.append(new_prompt)
                        adv_extras.append(extra)

                    # Map this mixed adversarial group to both YES and NO teacher indices
                    adv_teacher_index_groups.append([yes_indices[i_yes], no_indices[i_no]])

            return (
                adv_prompts,
                adv_extras,
                adv_teacher_index_groups,
            )

        # Default behavior: build from single teacher chain (no mixing)
        for i, (t_prompt, s_prompt, extra, prev_r, tgenerated) in enumerate(
            zip(
                combined_all_teacher_prompts,
                combined_all_student_prompts,
                combined_extras,
                extracted_reasonings,
                teacher_generated,
            )
        ):
            new_prompt = create_student_prompt(
                extra["dialogue"], bos_token=bos_token, previous_reasoning=prev_r, cfg=self.cfg
            )
            new_extra = extra

            for _ in range(self.cfg.adv_n_samples_per_prompt):
                adv_prompts.append(new_prompt)
                adv_extras.append(new_extra)

            # Non-mixed: reward applies back to this single teacher index
            adv_teacher_index_groups.append([i])

        return (
            adv_prompts,
            adv_extras,
            adv_teacher_index_groups,
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


    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]], packing_max_len: int
    ):
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            seq_offset = 0
            seq_index = 0
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
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
        ):
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            return seq_offset + total_len, seq_index + 1

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # make packed sequences
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens)
        ):
            if seq_offset + total_len < packing_max_len:
                seq_offset, seq_index = _accumulate(
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
                )
            elif seq_offset + total_len == packing_max_len:
                seq_offset, seq_index = _accumulate(
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
                )
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
            elif seq_offset + total_len > packing_max_len:
                if seq_offset > 0:
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                    ) = _new_instance()
                    seq_offset, seq_index = _accumulate(
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
                    )

        if seq_offset > 0:
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)

        return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards

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

    async def _sync_teacher_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.teacher_model.async_run_method("_broadcast_to_vllm_cudaipc", self.vllm_engines)
        else:
            await self.teacher_model.async_run_method("_broadcast_to_vllm", self.vllm_engines)

    async def _major_sync_policy_weights_to_vllm(self):
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()
            await self._sync_policy_weights_to_vllm()
            await self.policy_model.offload_to_cpu()
        else:
            await self._sync_policy_weights_to_vllm()

    async def _major_sync_teacher_weights_to_vllm(self):
        if self.cfg.colocate_all:
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
        self.vllm_engines, self._vllm_pg_handles = create_vllm_engines(
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
            return_pg_handles=True,
        )

        # 3) Re-initialize the comm groups on policy/teacher models for these engines
        if role == "student":
            await self.policy_model.async_run_method("_init_vllm_engines_actor_group", self.vllm_engines)
        elif role == "teacher":
            await self.teacher_model.async_run_method("_init_teacher_vllm_engines_actor_group", self.vllm_engines)
    async def _ensure_vllm_role(self, role: str):
        """Ensure vLLM engines are created for the requested role ('student'|'teacher').

        When cfg.vllm_recreate_on_switch is False, this is a no-op.
        When True and role changes, we recreate engines with the appropriate model
        and sync corresponding weights to the engines.
        """
        if not self.cfg.vllm_recreate_on_switch:
            return
        if role == self._vllm_current_role:
            return

        # Choose model for the role
        pretrain_model = self.cfg.teacher_pretrain if role == "teacher" else self.cfg.pretrain

        # Recreate and re-sync
        await self._recreate_vllm_engines(pretrain=pretrain_model, role=role)
        await self._backload_vllm_engines()

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
