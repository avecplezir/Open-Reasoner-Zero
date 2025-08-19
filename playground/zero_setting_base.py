from typing import List

from jinja2 import Template

from orz.ppo import PromptDataset


PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. If the question can be answered with 'yes' or 'no', your answer must be 'yes' or 'no'.
This is the problem:
{{prompt}}
"""

TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User gives a question and its final answer. The Assistant reconstructs the reasoning process in the mind that leads to this asnwer, and then recstate the User's final answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}} The final answer is {{answer}}. 
Assistant: <think>\
"""

STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""

def create_teacher_prompt_from_answer(dialogue: List, answer: str = "", bos_token: str = ""):
    """Create teacher prompt with student answer."""
    teacher_prompt_template_jinja = TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA

    prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA

    assert len(dialogue) == 2, "dialogue must contain 2 items"

    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
    teacher_prompt_template = Template(teacher_prompt_template_jinja)

    teacher_prompt_answer = teacher_prompt_template.render(
        bos_token=bos_token,
        prompt=prompt_instruction,
        answer=answer
    )

    return teacher_prompt_answer

class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):
        prompt_template_jinja = STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA

        prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA

        assert len(dialogue) == 2, "dialogue must contain 2 items"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        # Create teacher prompt with ground truth answer
        teacher_prompt = self.create_teacher_prompt(dialogue)
        
        extra = {
            "answer": dialogue[1]["ground_truth"]["value"],
            "teacher_prompt_yes": teacher_prompt[0],
            "teacher_prompt_no": teacher_prompt[1],
            "dialogue": dialogue,

        }

        return prompt, extra

    def create_teacher_prompt(self, dialogue: List):
        """Create teacher prompt with ground truth answer included."""
        teacher_prompt_template_jinja = TEACHER_PROMPT_INSTRUCTION_TEMPLATE_JNJA

        prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA

        assert len(dialogue) == 2, "dialogue must contain 2 items"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
        teacher_prompt_template = Template(teacher_prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        
        answer = dialogue[1]["ground_truth"]["value"]
        teacher_prompt_yes = teacher_prompt_template.render(
            bos_token=bos_token, 
            prompt=prompt_instruction,
            answer='yes'
        )

        teacher_prompt_no = teacher_prompt_template.render(
            bos_token=bos_token,
            prompt=prompt_instruction,
            answer='no'
        )

        return teacher_prompt_yes, teacher_prompt_no


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        prompt_template_jinja = STUDENT_PROMPT_INSTRUCTION_TEMPLATE_JNJA
        prompt_instruction_template_jinja = PROMPT_INSTRUCTION_TEMPLATE_JNJA

        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue["prompt"][0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
