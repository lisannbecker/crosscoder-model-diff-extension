import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ─── Config ────────────────────────────────────────────────────────────────────

TOTAL_TOKENS = 1_000_000
MEMMAP_PATH  = "owt-1m-tokenized-gpt2.dat"
PT_PATH      = "owt-1m-tokenized-gpt2.pt"


# ─── Load & tweak the GPT-2 tokenizer ─────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
# override the default max length so we never hit a warning
tokenizer.model_max_length = 10**12  

# ─── Stream & tokenize OpenWebText ─────────────────────────────────────────────

ds = load_dataset("openwebtext", split="train", streaming=True)

fp = np.memmap(MEMMAP_PATH, dtype=np.int32, mode="w+", shape=(TOTAL_TOKENS,))
idx = 0
print("Tokenizing into GPT-2 vocab…")
for ex in ds:
    ids = tokenizer.encode(ex["text"], add_special_tokens=False)
    n   = min(len(ids), TOTAL_TOKENS - idx)
    fp[idx : idx + n] = np.array(ids[:n], dtype=np.int32)
    idx += n
    if idx >= TOTAL_TOKENS:
        break
fp.flush()

# ─── Copy out & save as a torch LongTensor ────────────────────────────────────

# copy the slice we actually filled, then convert to long
arr = fp[:idx].copy()
tokens = torch.from_numpy(arr).long()
torch.save(tokens, PT_PATH)
print(f"Saved {idx:,} tokens → {PT_PATH}")


os.remove(MEMMAP_PATH)
