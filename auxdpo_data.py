"""
DPO Data Loading for MLX
========================

Loads preference pairs in the format:
  {"chosen": [{"role": "user", ...}, {"role": "assistant", ...}],
   "rejected": [{"role": "user", ...}, {"role": "assistant", ...}]}

Tokenizes using the model's chat template, computes masks that isolate
only the assistant response tokens (prompt tokens are masked out of
the loss computation).
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import mlx.core as mx
import numpy as np


class DPODataset:
    """
    Dataset for DPO preference pairs.

    Each item returns:
      (chosen_tokens, rejected_tokens, chosen_mask, rejected_mask)

    Masks are 1 for assistant response tokens, 0 for prompt/padding.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048,
    ):
        self._data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[list, list, list, list]:
        item = self._data[idx]
        chosen_messages = item["chosen"]
        rejected_messages = item["rejected"]

        chosen_tokens, chosen_mask = self._tokenize_with_mask(chosen_messages)
        rejected_tokens, rejected_mask = self._tokenize_with_mask(rejected_messages)

        return chosen_tokens, rejected_tokens, chosen_mask, rejected_mask

    def _tokenize_with_mask(self, messages: List[Dict]) -> Tuple[list, list]:
        """
        Tokenize a conversation and create a mask that is 1 only for
        assistant response tokens. The prompt portion is masked to 0.
        """
        # Full conversation tokens
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            return_dict=False,
        )

        # Prompt-only tokens (everything except the last assistant turn)
        # to determine the offset where the response begins
        prompt_messages = messages[:-1]
        add_gen_prompt = messages[-1].get("role") == "assistant"
        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=add_gen_prompt,
            return_dict=False,
        )

        prompt_len = len(prompt_tokens)
        full_len = min(len(full_tokens), self.max_length)

        # Truncate
        tokens = full_tokens[:full_len]

        # Mask: 1 for response tokens, 0 for prompt and padding
        mask = [0] * min(prompt_len, full_len) + [1] * max(0, full_len - prompt_len)

        return tokens, mask

    def itemlen(self, idx: int) -> int:
        """Return approximate length for sorting."""
        item = self._data[idx]
        # Rough estimate without full tokenization
        chosen_text = " ".join(m.get("content", "") for m in item["chosen"])
        rejected_text = " ".join(m.get("content", "") for m in item["rejected"])
        return max(len(chosen_text), len(rejected_text))


def load_dpo_data(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DPODataset, Optional[DPODataset]]:
    """
    Load DPO pairs from a JSONL file and split into train/val.

    Expected format per line:
      {"chosen": [...], "rejected": [...]}

    Args:
        data_path: path to .jsonl file
        tokenizer: HF tokenizer with chat template
        max_length: maximum sequence length
        val_split: fraction held out for validation (0 = no val set)
        seed: random seed for split

    Returns:
        (train_dataset, val_dataset) or (train_dataset, None)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"DPO data file not found: {data_path}")

    data = []
    with open(data_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed JSON at line {line_num}: {e}")
                continue

            if "chosen" not in item or "rejected" not in item:
                print(f"[WARNING] Skipping line {line_num}: missing 'chosen' or 'rejected'")
                continue

            data.append(item)

    print(f"Loaded {len(data)} DPO pairs from {data_path}")

    if val_split > 0 and len(data) > 1:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(data))
        n_val = max(1, int(len(data) * val_split))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        return (
            DPODataset(train_data, tokenizer, max_length),
            DPODataset(val_data, tokenizer, max_length),
        )
    else:
        return DPODataset(data, tokenizer, max_length), None


def collate_dpo_batch(
    batch: List[Tuple],
    pad_token_id: int = 0,
) -> Dict[str, mx.array]:
    """
    Collate a list of DPO examples into padded batch tensors.

    Args:
        batch: list of (chosen_tokens, rejected_tokens, chosen_mask, rejected_mask)
        pad_token_id: token id used for padding

    Returns:
        dict with keys: chosen_ids, rejected_ids, chosen_mask, rejected_mask
    """
    chosen_tokens_list = [b[0] for b in batch]
    rejected_tokens_list = [b[1] for b in batch]
    chosen_mask_list = [b[2] for b in batch]
    rejected_mask_list = [b[3] for b in batch]

    # Find max length across all sequences in the batch
    max_len = max(
        max(len(t) for t in chosen_tokens_list),
        max(len(t) for t in rejected_tokens_list),
    )

    # Pad to nearest multiple of 32
    pad_to = 32
    max_len = pad_to * ((max_len + pad_to - 1) // pad_to)

    batch_size = len(batch)

    def pad_sequences(token_lists, mask_lists):
        ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int32)
        masks = np.zeros((batch_size, max_len), dtype=np.float32)
        for i, (toks, msk) in enumerate(zip(token_lists, mask_lists)):
            length = min(len(toks), max_len)
            ids[i, :length] = toks[:length]
            masks[i, :length] = msk[:length]
        return mx.array(ids), mx.array(masks)

    chosen_ids, chosen_mask = pad_sequences(chosen_tokens_list, chosen_mask_list)
    rejected_ids, rejected_mask = pad_sequences(rejected_tokens_list, rejected_mask_list)

    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


def iterate_dpo_batches(
    dataset: DPODataset,
    batch_size: int = 1,
    max_seq_length: int = 2048,
    loop: bool = False,
    seed: Optional[int] = None,
):
    """
    Yield batches of DPO pairs, sorted by length for efficiency.

    Args:
        dataset: DPODataset instance
        batch_size: examples per batch
        max_seq_length: truncation length
        loop: whether to loop indefinitely
        seed: random seed for shuffling

    Yields:
        dict with chosen_ids, rejected_ids, chosen_mask, rejected_mask
    """
    n = len(dataset)
    if n < batch_size:
        raise ValueError(
            f"Dataset has {n} examples but batch_size={batch_size}"
        )

    # Sort by approximate length
    idx = sorted(range(n), key=lambda i: dataset.itemlen(i))

    # Create batch indices
    batch_idx = [idx[i:i + batch_size] for i in range(0, n - batch_size + 1, batch_size)]

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    while True:
        order = rng.permutation(len(batch_idx))
        for bi in order:
            examples = [dataset[j] for j in batch_idx[bi]]
            pad_id = dataset.tokenizer.pad_token_id or 0
            yield collate_dpo_batch(examples, pad_token_id=pad_id)

        if not loop:
            break
