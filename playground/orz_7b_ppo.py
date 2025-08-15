"""
Qwen2.5-7B base model + ppo

debug running command in single node:
DEBUG_MODE=True python -m playground.orz_7b_ppo

Multi-node Training:

on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`
then on master node, run `python -m playground.orz_7b_ppo`

"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import islice, zip_longest
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
import ray
import torch
import wandb
from loguru import logger
from omegaconf.listconfig import ListConfig
from typing_extensions import override

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import RayPPOTrainer
from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.utils import check_reflection_pattern
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
executor = ThreadPoolExecutor(max_workers=64)


def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    total_num_nodes: int = 32 if not DEBUG_MODE else 8

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
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B" # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"orz_ckpt/{file_name}"
    save_path: str = f"orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "data/orz_math_57k_collected.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
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
    rollout_batch_size: int = 128 if not DEBUG_MODE else 16
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 128

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

    enable_eval: bool = True
    eval_interval: int = 10

    # generate related settings
    packing_max_len: int = 16384
    generate_max_len: int = 8000  # TODO: change to larger later
    max_len: int = 8192  # TODO: change to larger later
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = False

    gpu_memory_utilization: float = 0.75 if use_grpo else 0.7 if not DEBUG_MODE else 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0


class CustomRewardTrainer(RayPPOTrainer):
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        # make log metrics
        scores = []
        teacher_scores = []
        responses = []
        avg_non_stop_count = 0
        pass_at_n_dict = defaultdict(list)
        teacher_pass_at_n_dict = defaultdict(list)
        num_tokens: List[int] = []

        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []
        for output in outputs:
            response = output["response"]
            # calculate repeat score for log
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)

        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )

        # Log reasoning chains, generated answers, and true answers to wandb
        if wandb.run is not None:
            # Log sample reasoning chains and answers
            reasoning_examples = []
            for i in range(min(5, len(outputs))):  # Log up to 5 examples
                true_answer = extras[i].get("answer", "N/A") if i < len(extras) else "N/A"
                reasoning_examples.append({
                    "prompt": prompts[i],
                    "reasoning_chain": outputs[i]['response'],
                    "generated_answer": outputs[i]['final_answer'], 
                    "true_answer": true_answer,
                    "is_correct": outputs[i]['iscorrect'],
                    'teacher_iscorrect': outputs[i]['teacher_iscorrect'],
                    "stop_reason": outputs[i]['stop_reason']
                })

            wandb.log({
                "reasoning_examples": wandb.Table(
                    columns=["prompt", "reasoning_chain", "generated_answer", "true_answer", "is_correct", "teacher_iscorrect", "stop_reason"],
                    data=[[ex["prompt"], ex["reasoning_chain"], ex["generated_answer"], ex["true_answer"], ex["is_correct"], ex['teacher_iscorrect'], ex["stop_reason"]] for ex in reasoning_examples]
                )
            }, step=self.global_step)

        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]
            teacher_iscorrect = output["teacher_iscorrect"]
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score
            # only correct and stoped response can aquire reward
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
                teacher_score = 1.0 if teacher_iscorrect else 0.0
            else:
                avg_non_stop_count += 1
                score = 0.0
                teacher_score = 0.0
            scores.append(score)
            teacher_scores.append(teacher_score)

            # calculate pass@n
            pass_at_n_dict[prompt].append(scores[-1])
            teacher_pass_at_n_dict[prompt].append(teacher_scores[-1])
            # log num_tokens
            num_tokens.append(response_token)

        # must before grpo, for grpo will change scores
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        initial_scores = copy.deepcopy(scores)
        initial_teacher_scores = copy.deepcopy(teacher_scores)
        # GRPO
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            self.writer.add_scalar("grpo_teacher_raw_reward", np.mean(teacher_scores), self.global_step)
            # grpo reward normalization
            for i, prompt in enumerate(prompts):
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std
                if not self.cfg.grpo_normalize_only_at_trainer:
                    teacher_scores[i] -= np.mean(teacher_pass_at_n_dict[prompt])
                    if teacher_std := np.std(teacher_pass_at_n_dict[prompt]) > 0:
                        teacher_scores[i] /= teacher_std

        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_repeat_score": sum(repeat_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_teacher_pass_at_n": sum(1 for v in teacher_pass_at_n_dict.values() if np.sum(v) > 0) / len(teacher_pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # make histogram for correct and incorrect response length
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # make a pre-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        teacher_score_tensors = []
        for score, teacher_score, output_token in zip(scores, teacher_scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            teacher_score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
                teacher_score_tensor[-1] = teacher_score
            score_tensors.append(score_tensor)
            teacher_score_tensors.append(teacher_score_tensor)

        # rm empty response
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        res_teacher_score_tensors = []
        res_indices = []
        for prompt, response, output, score_tensor, teacher_score_tensor in zip(prompts, responses, outputs, score_tensors, teacher_score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)
                res_teacher_score_tensors.append(teacher_score_tensor)
                # Extract indices from output, defaulting to None if not present
                begin_idx = output.get('answer_begin_idx', None)
                end_idx = output.get('answer_end_idx', None)
                res_indices.append((begin_idx, end_idx))

        return res_prompts, res_responses, res_score_tensors, res_teacher_score_tensors, res_indices, np.array(initial_scores), np.array(initial_teacher_scores)

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        # read sampling params from self.cfg

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str], tokenizer) -> List[dict]:
            # pattern = re.compile(r"(\\boxed{.*})")
            # pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            pattern = re.compile(r"<answer>(.*)</answer>", re.DOTALL)
            results = []
            for response in responses:
                matches = re.findall(pattern, response)
                final_answer = matches[-1] if matches else ""
                
                # Compute begin and end indices of final answer in tokenized response
                answer_begin_idx, answer_end_idx = None, None
                if final_answer:
                    # Find the position of <answer> and </answer> tags
                    answer_start = response.find("<answer>")
                    answer_end = response.find("</answer>")
                    if answer_start != -1 and answer_end != -1:
                        # Extract just the content between tags
                        answer_content_start = answer_start + len("<answer>")
                        answer_content = response[answer_content_start:answer_end]
                        
                        # Tokenize the full response to get token indices
                        tokenized_full = tokenizer.encode(response, add_special_tokens=False)
                        
                        # Find where the answer content starts and ends by tokenizing segments
                        prefix = response[:answer_content_start]
                        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                        answer_begin_idx = len(prefix_tokens)
                        
                        # Tokenize prefix + answer content to find end
                        prefix_plus_answer = response[:answer_end] 
                        prefix_plus_answer_tokens = tokenizer.encode(prefix_plus_answer, add_special_tokens=False)
                        answer_end_idx = len(prefix_plus_answer_tokens)
                        
                        # Verification: detokenize the extracted tokens back to text
                        # if answer_begin_idx is not None and answer_end_idx is not None:
                        #     answer_tokens = tokenized_full[answer_begin_idx:answer_end_idx]
                        #     detokenized_answer = tokenizer.decode(answer_tokens, skip_special_tokens=False)
                        #     logger.info(f"Original final_answer: '{final_answer}' Answer content: '{answer_content}' Detokenized from indices [{answer_begin_idx}:{answer_end_idx}]: '{detokenized_answer}'")

                results.append({
                    "final_answer": final_answer,
                    "answer_begin_idx": answer_begin_idx,
                    "answer_end_idx": answer_end_idx
                })
            return results

        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        # 直接从context中提取最终结果
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch, self.tokenizer))
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answer_items = [item for batch in batched_results for item in batch]

        # 判断对错
        global executor
        equal_tasks = []
        for extra, final_answer_item in zip(extras, final_answer_items):
            equal_tasks.append(is_equal(solution2answer(extra["answer"]), solution2answer(final_answer_item['final_answer']), executor))
        equal_results = await asyncio.gather(*equal_tasks)

        equal_teacher_tasks = []
        for extra, final_answer_item in zip(extras, final_answer_items):
            equal_teacher_tasks.append(is_equal(solution2answer(extra["teacher_answer"]), solution2answer(final_answer_item['final_answer']), executor))
        equal_teacher_results = await asyncio.gather(*equal_teacher_tasks)

        results = []
        for extra, response, final_answer_item, stop_reason, iscorrect, teacher_iscorrect in zip(
            extras, responses, final_answer_items, stop_reasons, equal_results, equal_teacher_results
        ):
            results.append(
                dict(
                    response=response,
                    iscorrect=iscorrect,
                    teacher_iscorrect=teacher_iscorrect,
                    stop_reason=stop_reason,
                    final_answer=final_answer_item['final_answer'],
                    answer_begin_idx=final_answer_item['answer_begin_idx'],
                    answer_end_idx=final_answer_item['answer_end_idx'],
                )
            )

        return results

    @override
    async def eval(self):
        logger.info("Start evaluating on val set")
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        from torch.utils.data import DataLoader

        dataset = self.eval_dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

        output_for_save = []
        log_dict = defaultdict(float)
        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])
            outputs = []
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                    )
                )
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            final_answers = []
            # pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            pattern = re.compile(r"<answer>(.*)</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)
                prefix_response = solution2answer(final_answer)

                iscorrect = await is_equal(label, prefix_response, executor)
                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # get all file_names from self.cfg.eval_prompt_data
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
        ]
        for file_name in all_file_names:
            log_dict[f"{file_name}/response_len_in_char"] = (
                log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            )
            log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/correct")
            log_dict.pop(f"{file_name}/total")
        # calculate average accuracy
        log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            all_file_names
        )

        dump_file_name = f"eval_output_iter{self.global_step}"
        # join all acc from all_file_names
        for file_name in all_file_names:
            dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
        dump_file_name += ".jsonl"
        # dump as jsonl
        with open(
            os.path.join(
                self.cfg.save_path,
                dump_file_name,
            ),
            "w",
        ) as f:
            for item in output_for_save:
                f.write(
                    json.dumps(item, ensure_ascii=False) + "\n",
                )

        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)
        for k, v in log_dict.items():
            self.writer.add_scalar(f"evals/{k}", v, self.global_step)

        # Log evaluation results to wandb
        if wandb.run is not None:
            # Log evaluation examples
            eval_examples = []
            for i, item in enumerate(output_for_save[:5]):  # Log up to 5 examples
                eval_examples.append({
                    "prompt": item["prompt"],
                    "reasoning_chain": item["output"],
                    "generated_answer": item["final_answer"],
                    "true_answer": item["answer"], 
                    "is_correct": item["iscorrect"]
                })
            
            wandb.log({
                "eval_reasoning_examples": wandb.Table(
                    columns=["prompt", "reasoning_chain", "generated_answer", "true_answer", "is_correct"],
                    data=[[ex["prompt"], ex["reasoning_chain"], ex["generated_answer"], ex["true_answer"], ex["is_correct"]] for ex in eval_examples]
                )
            }, step=self.global_step)


class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                for loaded_data_item in loaded_data:
                    # only keep file name, without suffix
                    loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
