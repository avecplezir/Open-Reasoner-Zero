from typing import List, Optional, Tuple
from collections import defaultdict, deque
import random
import re
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


# Teacher variant (SAY mode): explanation + answer, but only content wrapped in
# <say>...</say> is intended to be visible to the student. The rest of the
# reasoning inside <think> is internal and should not be exposed.
TEACHER_PROMPT_INSTRUCTION_SAY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question and its final answer. \ 
The Assistant reasons internally inside <think>...</think> and may include one or more <say>...</say> segments to mark \ 
brief student-visible statements. Only the text inside <say>...</say> will be visible to the student; all other text \ 
remains internal. End by restating the final answer in <answer>...</answer> tags. \ 
User: {{prompt}} The final answer is {{answer}}. 
Assistant: <think>\
"""

# SAY mode: explanation-only (no <answer> in the output). We will append the
# provided answer programmatically after generation. The teacher should still
# place student-visible statements inside <say>...</say> blocks.
TEACHER_PROMPT_EXPLAIN_ONLY_SAY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question and its final answer. \ 
The Assistant reconstructs only the internal reasoning inside <think>...</think>. Any student-visible statements must be \ 
explicitly wrapped inside <say>...</say> blocks. Output only the content within <think>...</think> and DO NOT output the \ 
<answer> tag. User: {{prompt}} The final answer is {{answer}}. 
Assistant: <think>\
"""

# SAY mode: correctness-guided variants
TEACHER_PROMPT_CORRECT_ONLY_SAY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question. The Assistant must solve it correctly. \ 
The Assistant first reasons internally inside <think>...</think>, using <say>...</say> blocks for any student-visible \ 
statements. Finish with a correct final answer inside <answer>...</answer> tags. \ 
User: {{prompt}} 
Assistant: <think>\
"""

TEACHER_PROMPT_INCORRECT_ONLY_SAY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question. The Assistant must provide a plausible \ 
but incorrect answer. The Assistant reasons internally inside <think>...</think>, using <say>...</say> blocks for any \ 
student-visible statements, and ends with an incorrect final answer inside <answer>...</answer> tags. Avoid trivial mistakes; \ 
the solution should be coherent but lead to a wrong final answer. \ 
User: {{prompt}} 
Assistant: <think>\
"""


# Teacher variant: correctness-guided without revealing the answer.
# We ask the teacher to solve the question and end with a correct or incorrect
# final answer depending on the instruction. This is used by the
# `correct_incorrect_generate` augmentation strategy.
TEACHER_PROMPT_CORRECT_ONLY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question. The Assistant must solve it correctly. \
The Assistant first reasons in the mind and then provides the User with a correct final answer. \
The reasoning process is enclosed within <think> </think> and the final answer must be enclosed within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
User: {{prompt}} 
Assistant: <think>\
"""

TEACHER_PROMPT_INCORRECT_ONLY_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question. The Assistant must provide a plausible but incorrect answer. \
The Assistant first reasons in the mind and then provides the User with an incorrect final answer. Avoid trivial mistakes; the solution should be coherent but lead to a wrong final answer. \
The reasoning process is enclosed within <think> </think> and the final answer must be enclosed within <answer> </answer> tags, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
User: {{prompt}} 
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

    # Choose base template. If the augmentation strategy requests correctness-guided
    # generation without revealing the answer, switch to the corresponding templates.
    if cfg.augment_strategy == "correct_incorrect" and cfg.teacher_no_prompt_answer:
        assert is_correct is not None, "is_correct must be provided for correctness-guided teacher prompts"
        assert not cfg.teacher_explain_only, "correct_incorrect requires teacher_explain_only=False to emit <answer> when teacher_no_prompt_answer is True"
        if cfg.teacher_use_say_operator:
            teacher_prompt_template_jinja = (
                TEACHER_PROMPT_CORRECT_ONLY_SAY_TEMPLATE_JNJA
                if is_correct
                else TEACHER_PROMPT_INCORRECT_ONLY_SAY_TEMPLATE_JNJA
            )
        else:
            teacher_prompt_template_jinja = (
                TEACHER_PROMPT_CORRECT_ONLY_TEMPLATE_JNJA if is_correct else TEACHER_PROMPT_INCORRECT_ONLY_TEMPLATE_JNJA
            )
    else:
        if cfg.teacher_use_say_operator:
            teacher_prompt_template_jinja = (
                TEACHER_PROMPT_EXPLAIN_ONLY_SAY_TEMPLATE_JNJA
                if cfg.teacher_explain_only
                else TEACHER_PROMPT_INSTRUCTION_SAY_TEMPLATE_JNJA
            )
        else:
            teacher_prompt_template_jinja = (
                TEACHER_PROMPT_EXPLAIN_ONLY_TEMPLATE_JNJA
                if cfg.teacher_explain_only
                else TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA
            )

    if not eval:
        assert isinstance(dialogue, (list, str)), "dialogue must be a list or pre-rendered string"
        if isinstance(dialogue, list):
            assert len(dialogue) == 2, "dialogue must contain 2 items"

    # Base prompt text (question). Allow passing a pre-rendered string
    if isinstance(dialogue, str):
        prompt = dialogue
    else:
        prompt = dialogue["prompt"][0]["value"] if eval else dialogue[0]["value"]

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
    def __init__(self, maxlen: int = 8):
        # key -> {label: deque}
        self._buf = defaultdict(
            lambda: {
                "yes": deque(maxlen=maxlen),
                "no": deque(maxlen=maxlen),
                "correct": deque(maxlen=maxlen),
                "incorrect": deque(maxlen=maxlen),
            }
        )
        self.sample_last = False

    def add(self, key: str, attempt: str, label: str) -> bool:
        """Add an attempt under a label.

        Accepts label in {yes, no} (normalized via solution2answer) or
        {correct, incorrect} directly. Returns True if added.
        """
        if label is None:
            return False
        raw = str(label).strip().lower()
        mapped = solution2answer(raw).strip().lower()
        if mapped in ("yes", "no"):
            lab = mapped
        elif raw in ("correct", "incorrect"):
            lab = raw
        else:
            return False
        dq = self._buf[key][lab]
        dq.append(attempt)
        return True

    def add_correctness(self, key: str, attempt: str, is_correct: bool) -> bool:
        """Convenience method to add using a boolean correctness label."""
        lab = "correct" if bool(is_correct) else "incorrect"
        dq = self._buf[key][lab]
        dq.append(attempt)
        return True

    def sample_by_labels(self, key: str, labels: Tuple[str, str], k: int = 1) -> Tuple[List[str], List[str]]:
        """Sample up to k attempts for each of the two provided labels.

        Labels should be present in the buffer (e.g., ("yes","no") or ("correct","incorrect")).
        Respects `sample_last` to either take last-k or random-k.
        """
        if k <= 0:
            return [], []
        a_label, b_label = labels
        a_list = list(self._buf.get(key, {}).get(a_label, deque()))
        b_list = list(self._buf.get(key, {}).get(b_label, deque()))
        if not self.sample_last:
            a = random.sample(a_list, k=min(k, len(a_list))) if a_list else []
            b = random.sample(b_list, k=min(k, len(b_list))) if b_list else []
        else:
            a = a_list[-k:] if a_list else []
            b = b_list[-k:] if b_list else []
        return a, b

    def sample(self, key: str, k: int = 1) -> Tuple[List[str], List[str]]:
        """Backward-compatible yes/no sampling."""
        return self.sample_by_labels(key, ("yes", "no"), k)



