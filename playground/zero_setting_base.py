from typing import List, Optional

from jinja2 import Template

from orz.ppo import PromptDataset


# Base prompt instruction template used in all variants
# PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. If the question can be answered with 'yes' or 'no', your answer must be 'yes' or 'no'.
# This is the problem:
# {{prompt}}
# """

# # student variant: explanation + answer (both <think> and <answer> in the output)
# STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """

# STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant may either: (1) reason from scratch; or (2) examine any previously provided reasoning and continue it. \
# If prior reasoning is provided, continue it to arrive at the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>{{previous_reasoning}}\
# """

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

STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant may either: (1) reason from scratch; or (2) examine any previously provided reasoning and continue it. \
If prior reasoning is provided, continue it to arrive at the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>{{previous_reasoning}}\
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
    template: str = "default",
    eval: bool = False,
) -> str:
    """Create a student prompt using a selectable template.

    Args:
        dialogue: Two-message list (user question + ground truth/metadata entry).
        bos_token: Model BOS token to prefix if present.
        previous_reasoning: Optional content of a prior <think>...</think> block.
        template: "default" or "continue".

    Returns:
        Rendered student prompt string.
    """

    if not eval:
        assert len(dialogue) == 2, "dialogue must contain 2 items"

    prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA
    if template == "continue":
        prompt_template_jinja = STUDENT_PROMPT_INSTRUCTION_CONTINUE_TEMPLATE_JNJA
    else:
        prompt_template_jinja = STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA

    prompt = dialogue["prompt"][0]["value"] if eval else dialogue[0]["value"]
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=prompt)
    prompt_template = Template(prompt_template_jinja)

    if template == "continue":
        rendered = prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction,
            previous_reasoning=previous_reasoning or "",
        )

    elif template == "default":
        rendered = prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction,
        )
    else:
        raise ValueError(f"Unknown template: {template}")

    return rendered


def create_student_adv_prompt(dialogue: List, bos_token: str = "", previous_reasoning: str = "" ) -> str:
    """Backward-compatible helper for adversarial continuation prompts."""
    return create_student_prompt(
        dialogue,
        bos_token=bos_token,
        previous_reasoning=previous_reasoning,
        template="continue",
    )


def create_teacher_prompt_from_answer(
    dialogue: List,
    answer: str = "",
    bos_token: str = "",
    *,
    explain_only: bool = False,
):
    """Create a teacher prompt from a dialogue and provided answer.

    When explain_only is True, the template instructs the assistant to output only
    the reasoning chain inside <think>...</think>, without emitting an <answer> tag
    (the caller may append the answer programmatically). Otherwise, the template
    asks for both reasoning and the final <answer>.
    """
    teacher_prompt_template_jinja = TEACHER_PROMPT_EXPLAIN_ONLY_TEMPLATE_JNJA if explain_only else TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA

    prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA

    assert len(dialogue) == 2, "dialogue must contain 2 items"

    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
    teacher_prompt_template = Template(teacher_prompt_template_jinja)

    teacher_prompt_answer = teacher_prompt_template.render(
        bos_token=bos_token,
        prompt=prompt_instruction,
        answer=answer,
    )

    return teacher_prompt_answer


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        # Optional flag to control teacher prompt style
        self.student_prompt_template = kwargs.pop("student_prompt_template", None)
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):

        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        # Resolve template: auto -> default for dataset usage
        prompt = create_student_prompt(dialogue, bos_token=bos_token, template=self.student_prompt_template)

        extra = {
            "answer": dialogue[1]["ground_truth"]["value"],
            "dialogue": dialogue,
        }

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        self.student_prompt_template = kwargs.pop("student_prompt_template", None)
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

        prompt = create_student_prompt(dialogue, bos_token=bos_token, template=self.student_prompt_template, eval=True)

        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
