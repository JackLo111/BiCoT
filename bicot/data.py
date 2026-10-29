import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IGNORE_TOKEN_ID = -100


class GSM8KCoTDataset(Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 2048,
        cache_dir: str = "./cache",
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

        self.anchor_pattern = re.compile(r"\b\d+\.?\d*\b")

        self.separator_ids = self.tokenizer.encode("\n####", add_special_tokens=False)
        if len(self.separator_ids) == 0:
            self.separator_ids = self.tokenizer.encode("####", add_special_tokens=False)

        self.data = self._load_and_process_data()

    def _load_and_process_data(self) -> list[dict[str, Any]]:
        cache_file = f"{self.cache_dir}/gsm8k_{self.split}_robust_v5.json"

        if os.path.exists(cache_file):
            print(f"Loading processed data from cache: {cache_file}")
            with open(cache_file) as f:
                return json.load(f)

        print(f"Processing GSM8K {self.split} split (Robust ID Matching)...")
        split_name = "test" if self.split == "validation" else self.split
        raw_dataset = load_dataset("gsm8k", "main", split=split_name)

        processed_data = []
        for item in raw_dataset:
            processed_item = self._process_single_item(item)
            if processed_item:
                processed_data.append(processed_item)

        with open(cache_file, "w") as f:
            json.dump(processed_data, f, indent=2)
        return processed_data

    def _process_single_item(self, item: dict[str, str]) -> Optional[dict[str, Any]]:
        question = item["question"].strip()
        answer = item["answer"].strip()

        if "####" in answer:
            parts = answer.split("####")
            cot = parts[0].strip()
            answer_final = parts[-1].strip()
        else:
            return None

        return {"question": question, "cot_reasoning": cot, "final_answer": answer_final}

    def _find_subsequence(
        self, main_seq: list[int], sub_seq: list[int], start_index: int = 0
    ) -> int:
        n = len(main_seq)
        m = len(sub_seq)
        if m == 0:
            return -1

        for i in range(start_index, n - m + 1):
            if main_seq[i : i + m] == sub_seq:
                return i
        return -1

    def find_anchor_indices_with_offsets(self, text, input_ids, offset_mapping):
        matches = list(self.anchor_pattern.finditer(text))
        anchor_mask = torch.zeros(len(input_ids), dtype=torch.bool)

        match_idx = 0
        for i, (start, end) in enumerate(offset_mapping):
            if match_idx >= len(matches):
                break

            if start == end:
                continue

            m_start, m_end = matches[match_idx].span()

            if start < m_end and end > m_start:
                anchor_mask[i] = True

            if end >= m_end:
                match_idx += 1

        return anchor_mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Solve the math problem step by step.",
            },
            {"role": "user", "content": item["question"]},
            {
                "role": "assistant",
                "content": item["cot_reasoning"] + "\n#### " + item["final_answer"],
            },
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoding.input_ids[0]
        offset_mapping = encoding.offset_mapping[0]
        attention_mask = encoding.attention_mask[0]
        input_ids_list = input_ids.tolist()


        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        assistant_start_idx = len(prompt_ids)

        answer_start_idx = self._find_subsequence(
            input_ids_list, self.separator_ids, start_index=assistant_start_idx
        )
        if answer_start_idx == -1:
            answer_start_idx = len(input_ids_list)

        labels = input_ids.clone()
        labels[:assistant_start_idx] = IGNORE_TOKEN_ID

        anchor_mask = self.find_anchor_indices_with_offsets(
            full_text, input_ids_list, offset_mapping
        )

        anchor_mask[:assistant_start_idx] = False
        if answer_start_idx < len(input_ids):
            anchor_mask[answer_start_idx:] = False

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "anchor_mask": anchor_mask,
        }

    def __len__(self):
        return len(self.data)


@dataclass
class BiCoTDataCollator:
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        anchor_mask = [f["anchor_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        padded_anchor_masks = []

        for i, (lab, anc) in enumerate(zip(labels, anchor_mask)):
            pad_len = max_len - len(lab)

            if self.tokenizer.padding_side == "left":
                p_lab = torch.cat([torch.full((pad_len,), IGNORE_TOKEN_ID, dtype=lab.dtype), lab])
                p_anc = torch.cat([torch.full((pad_len,), False, dtype=anc.dtype), anc])
            else:
                p_lab = torch.cat([lab, torch.full((pad_len,), IGNORE_TOKEN_ID, dtype=lab.dtype)])
                p_anc = torch.cat([anc, torch.full((pad_len,), False, dtype=anc.dtype)])

            padded_labels.append(p_lab)
            padded_anchor_masks.append(p_anc)

        batch["labels"] = torch.stack(padded_labels)
        batch["anchor_mask"] = torch.stack(padded_anchor_masks)

        return batch


def get_gsm8k_dataset(tokenizer):
    train_ds = GSM8KCoTDataset(tokenizer, split="train")
    test_ds = GSM8KCoTDataset(tokenizer, split="test")
    return {"train": train_ds, "test": test_ds}
