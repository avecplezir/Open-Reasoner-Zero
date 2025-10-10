import multiprocessing


class PromptDataset:
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    # eot_text = EOT
    # bot_text = BOT

    # currently hack for llama 3.1
    start_header_text = "<|start_header_id|>"
    end_header_text = "<|end_header_id|>"
    eot_text = "<|eot_id|>"

    def __init__(
        self,
        dialogues,
        tokenizer: callable,
        max_length: int,
        strategy,
        pretrain_mode: bool = False,
        num_processors: int = 8,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        # Preprocess dialogues
        if num_processors < 2:
            self.dialogues = [self.process_dialogue(x) for x in dialogues]
        else:
            pool = multiprocessing.Pool(processes=num_processors)
            self.dialogues = pool.map(self.process_dialogue, dialogues)
            pool.close()
            pool.join()

    def process_dialogue(self, dialogue: dict):
        prompt_template = ""
        if self.tokenizer.bos_token_id is not None:
            prompt_template += f"{self.tokenizer.decode([self.tokenizer.bos_token_id])}"

        prompts = dialogue["prompt"]
        if prompts[-1]["role"] == "assistant":
            prompts = prompts[:-1]
        for message in prompts:
            prompt_template += f"{self.start_header_text}{message['role']}{self.end_header_text}\n{message['content']}{self.eot_text}\n"
        # append bot token
        prompt_template += f"{self.start_header_text}assistant{self.end_header_text}\n"

        extra = {key: value for key, value in dialogue.items() if key != "prompt"}

        return prompt_template, extra

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, extra in item_list:
            all_inputs.append((prompt, extra))
        return all_inputs

    def __getitem__(self, idx):
        inputs = self.dialogues[idx]
        return inputs

    def __len__(self):
        return len(self.dialogues)


class BalancedYesNoBatchSampler:
    """
    Batch sampler that yields approximately balanced yes/no batches.

    It inspects the dataset's processed `dialogues` list, where each item is
    a tuple `(prompt_text, extra)` and `extra` is expected to contain an
    answer-like field indicating yes/no, typically under keys like
    'answer' or 'final_answer'. Values can be strings ('yes'/'no', possibly
    boxed) or booleans.

    When there is a class imbalance, this sampler oversamples the minority
    class by cycling through its indices.
    """

    def __init__(self, dataset: PromptDataset, batch_size: int, drop_last: bool = False, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = int(seed)

        yes_idx, no_idx = [], []
        for i, (_, extra) in enumerate(self.dataset.dialogues):
            val = None
            if isinstance(extra, dict):
                # Prefer explicit 'answer', then 'final_answer'
                val = extra.get("answer", extra.get("final_answer"))
            # Normalize
            is_yes = None
            if isinstance(val, bool):
                is_yes = val
            elif isinstance(val, str):
                v = val.strip().lower()
                if "yes" in v:
                    is_yes = True
                elif "no" in v:
                    is_yes = False
            if is_yes is True:
                yes_idx.append(i)
            elif is_yes is False:
                no_idx.append(i)
            # If unknown, ignore for balancing; it will be filled via oversampling below if needed

        self._yes_idx = yes_idx
        self._no_idx = no_idx

    def __iter__(self):
        import random

        rng = random.Random(self.seed)
        yes = list(self._yes_idx)
        no = list(self._no_idx)
        # Fallback: if one class is empty, just yield sequential batches
        if len(yes) == 0 or len(no) == 0:
            all_idx = list(range(len(self.dataset)))
            rng.shuffle(all_idx)
            for i in range(0, len(all_idx), self.batch_size):
                batch = all_idx[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    break
                yield batch
            return

        rng.shuffle(yes)
        rng.shuffle(no)
        yi = ni = 0
        yN, nN = len(yes), len(no)

        half = self.batch_size // 2
        extra = self.batch_size - half  # when odd, give +1 to 'no' side for variety

        batches = 0
        # We produce at most ceil(max(yN, nN) / half) batches unless drop_last=False
        max_pairs = max((yN + half - 1) // max(1, half), (nN + half - 1) // max(1, half))
        while True:
            batch = []
            for _ in range(half):
                batch.append(yes[yi])
                yi = (yi + 1) % yN
            for _ in range(extra):
                batch.append(no[ni])
                ni = (ni + 1) % nN
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch
            batches += 1
            if self.drop_last:
                if batches >= max_pairs:
                    break
            else:
                # For non-drop_last, stop when we have covered at least one full cycle
                if batches >= max_pairs:
                    break

    def __len__(self):
        # Approximate number of batches per epoch
        yN, nN = max(1, len(self._yes_idx)), max(1, len(self._no_idx))
        half = max(1, self.batch_size // 2)
        max_pairs = max((yN + half - 1) // half, (nN + half - 1) // half)
        return max_pairs if self.drop_last else max_pairs
