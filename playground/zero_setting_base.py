from typing import List, Optional, Tuple
from collections import defaultdict, deque
import random
from jinja2 import Template
from loguru import logger

from orz.ppo import PromptDataset
from orz.ppo.tools.math_utils import solution2answer


# Base prompt instruction templates used in all variants
# Variant with explicit yes/no guidance (previously commented out)
PROMPT_INSTRUCTION_TEMPLATE_JNJA_YESNO = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. If the question can be answered with 'yes' or 'no', your answer must be 'yes' or 'no'.
This is the problem:
{{prompt}}
"""

PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. 
This is the problem:
{{prompt}}
"""

STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""

# STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant may either: (1) reason from scratch; or (2) examine any previously provided reasoning and continue it. \
# If prior reasoning is provided, continue it to arrive at the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>{{previous_reasoning}}\
# """

# STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA = """\
# {{bos_token}} You are given a problem and several candidate solutions. \
# Some candidates may be incorrect or contain errors. Aggregate the useful ideas and produce a single, high-quality solution. \
# Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy. \n \
# {{prompt}} \n \
# Candidate solutions (may contain mistakes): {{previous_reasoning}}\n \
# Now write a single improved solution. Provide clear reasoning and end with the final answer in <answer>...</answer> tags \
# """

STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA = """\
{{bos_token}} A conversation between User and Assistant. You are given a problem and several candidate solutions. \
Some candidates may be incorrect or contain errors. Aggregate the useful ideas and produce a single, high-quality solution. \
Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy. \n \
{{prompt}} \n \
Candidate solutions (may contain mistakes): {{previous_reasoning}}\n \
Now write a single improved solution. Provide clear reasoning  enclosed within <think> </think> and end with the final answer in <answer>...</answer> tags. \
Assistant: <think>
"""

# Teacher variant: explanation + answer (both <think> and <answer> in the output)
TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question and its final answer. The Assistant reconstructs the reasoning process in the mind that leads to this asnwer, and then recstate the User's final answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}} The final answer is {{answer}}. 
Assistant: <think>\
"""

# Teacher variant: explanation only (no <answer> in the output). We will append
# the provided answer programmatically after generation.
TEACHER_PROMPT_EXPLAIN_ONLY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question and its final answer. The Assistant reconstructs only the reasoning process in the mind that leads to this answer. \
Output only the reasoning process inside <think> </think> tags and DO NOT output the <answer> tag. User: {{prompt}} The final answer is {{answer}}. 
Assistant: <think>\
"""


# PROMPT_INSTRUCTION_TEMPLATE_JNJA_BOXED = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag. If the question can be answered with 'yes' or 'no', your final answer must be '\\boxed{yes}' or '\\boxed{no}'.
# This is the problem:
# {{prompt}}
# """


def create_student_prompt(
    dialogue: List,
    bos_token: str = "",
    previous_reasoning: Optional[str] = None,
    *,
    cfg=None,
    eval: bool = False,
) -> str:
    """Create a student prompt using configuration-driven settings.

    Args:
        dialogue: Two-message list (user question + ground truth/metadata entry) or
            an eval dict with keys "prompt", "final_answer", etc. when eval=True.
        bos_token: Model BOS token to prefix if present.
        previous_reasoning: Optional content of a prior <think>...</think> block.
        cfg: Experiment configuration. Uses cfg.general_propmt_yes_no and
            cfg.student_prompt_continuation.
        eval: Eval-mode flag for dataset wrappers that pass dict inputs.

    Returns:
        Rendered student prompt string.
    """

    if not eval:
        assert len(dialogue) == 2, "dialogue must contain 2 items"

    # Decide which instruction header to use (general vs yes/no)
    prompt_instruction_template_jinja = get_instruction_template_from_flags(cfg.general_propmt_yes_no)

    # Map to the student prompt body variant
    prompt_template_jinja = STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA if cfg.student_prompt_continuation and previous_reasoning else STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA

    if isinstance(dialogue, str):
        prompt = dialogue
    else:
        prompt = dialogue["prompt"][0]["value"] if eval else dialogue[0]["value"]

    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=prompt)
    prompt_template = Template(prompt_template_jinja)

    if cfg.student_prompt_continuation:
        rendered = prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction,
            previous_reasoning=previous_reasoning or "",
        )
    else:
        rendered = prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction,
        )

    return rendered

def get_instruction_template_from_flags(yesno: bool) -> str:
    """Return the instruction header template based on yes/no flag."""
    return PROMPT_INSTRUCTION_TEMPLATE_JNJA_YESNO if yesno else PROMPT_INSTRUCTION_TEMPLATE_JNJA


def create_teacher_prompt_from_answer(
    dialogue: List,
    answer: str = "",
    bos_token: str = "",
    *,
    cfg=None,
    is_correct: Optional[bool] = None,
    eval: bool = False,
):
    """Create a teacher prompt from a dialogue and provided answer using cfg.

    The configuration controls two aspects:
    - cfg.general_propmt_yes_no: mirrors yes/no guidance with student.
    - cfg.teacher_explain_only: when True, asks for only <think> without <answer>.
    """
    prompt_instruction_template_jinja = get_instruction_template_from_flags(cfg.general_propmt_yes_no)

    teacher_prompt_template_jinja = TEACHER_PROMPT_EXPLAIN_ONLY_TEMPLATE_JNJA if cfg.teacher_explain_only else TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA

    if not eval:
        assert isinstance(dialogue, (list, str)), "dialogue must be a list or pre-rendered string"
        if isinstance(dialogue, list):
            assert len(dialogue) == 2, "dialogue must contain 2 items"

    # Base prompt text (question). Allow passing a pre-rendered string
    if isinstance(dialogue, str):
        prompt = dialogue
    else:
        prompt = dialogue["prompt"][0]["value"] if eval else dialogue[0]["value"]

    # If requested, inject previous student attempts (YES/NO) sampled from
    # the in-memory FIFO buffer. We keep the full attempts including <answer>.
    if cfg.use_student_history:
        y_hist_list, n_hist_list = _HISTORY_BUFFER.sample(prompt, k=cfg.student_history_samples_per_label)
        if y_hist_list or n_hist_list:
            parts = ["Previous student attempt(s):"]
            if y_hist_list:
                parts.append("[YES]")
                for idx, y in enumerate(y_hist_list, 1):
                    parts.append(f"- {y}")
            if n_hist_list:
                parts.append("[NO]")
                for idx, n in enumerate(n_hist_list, 1):
                    parts.append(f"- {n}")
            prompt = f"{prompt}\n\n" + "\n".join(parts)

    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=prompt)
    teacher_prompt_template = Template(teacher_prompt_template_jinja)

    # Optionally add a role prefix to the beginning via bos_token
    bos_with_role = bos_token
    if cfg.teacher_add_role_prefix:
        if is_correct is True:
            role = "You are a teacher explaining a correct final answer. "
            # role = "You are a teacher. Convince the student that the following correct final answer is correct."
        elif is_correct is False:
            role = "You are a teacher explaining an incorrect final answer. "
        else:
            assert 0, f"is_correct {is_correct} must be True or False if teacher_add_role_prefix is True"
            # role = "You are a teacher explaining the provided final answer. "
        bos_with_role = f"{bos_token}{role}"

    teacher_prompt_answer = teacher_prompt_template.render(
        bos_token=bos_with_role,
        prompt=prompt_instruction,
        answer=answer,
    )

    return teacher_prompt_answer


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        # Optional flag to control teacher prompt style
        self.cfg = kwargs.pop("cfg", None)
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):

        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        # Resolve template: auto -> default for dataset usage
        prompt = create_student_prompt(dialogue, bos_token=bos_token, cfg=self.cfg)

        extra = {
            "answer": dialogue[1]["ground_truth"]["value"],
            "dialogue": dialogue,
        }

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs.pop("cfg", None)
        # self.student_prompt_template = kwargs.pop("student_prompt_template", None)
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):

        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        prompt = create_student_prompt(dialogue, bos_token=bos_token, cfg=self.cfg, eval=True)

        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"], "dialogue": dialogue}

        return prompt, extra


# -----------------------
# In-memory FIFO history
# -----------------------

class _StudentHistoryBuffer:
    def __init__(self, maxlen: int = 1):
        self._buf = defaultdict(lambda: {"yes": deque(maxlen=maxlen), "no": deque(maxlen=maxlen)})  # key -> {label: deque}
        self.sample_last = False

    def add(self, key: str, attempt: str, label: str) -> bool:
        label_norm = solution2answer(label).strip().lower()
        if label_norm not in ("yes", "no"):
            return False
        dq = self._buf[key][label_norm]
        dq.append(attempt)
        return True

    def sample(self, key: str, k: int = 1) -> Tuple[List[str], List[str]]:
        if k <= 0:
            return [], []
        if not self.sample_last:
            # Randomly sample up to k prior attempts from each label, if present
            yes_list = list(self._buf.get(key, {}).get("yes", deque()))
            no_list = list(self._buf.get(key, {}).get("no", deque()))
            y = random.sample(yes_list, k=min(k, len(yes_list))) if yes_list else []
            n = random.sample(no_list, k=min(k, len(no_list))) if no_list else []
            return y, n
        else:
            # Return only the last k attempts from each label, if present
            yes_list = self._buf.get(key, {}).get("yes", deque())
            no_list = self._buf.get(key, {}).get("no", deque())
            y = list(yes_list)[-k:] if yes_list else []
            n = list(no_list)[-k:] if no_list else []
            return y, n


_HISTORY_BUFFER = _StudentHistoryBuffer()
