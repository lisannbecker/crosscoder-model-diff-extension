import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# 1 M tokens, memmap path
TOTAL_TOKENS = 1_000_000
MEMMAP_PATH  = "owt-1m-tokenized-pythia-160m.dat"
PT_PATH      = "owt-1m-tokenized-pythia-160m.pt"

# remove old files if present
for p in (MEMMAP_PATH, PT_PATH):
    if os.path.exists(p):
        os.remove(p)

# load Pythia-160 M tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

# stream OpenWebText raw text << different to Gemma because I couldnt make pile and lmsys work
ds = load_dataset("openwebtext", split="train", streaming=True)

# prepare a 1 M–int32 memmap
fp = np.memmap(MEMMAP_PATH, dtype=np.int32, mode="w+", shape=(TOTAL_TOKENS,))

print("Retokenizing into Pythia vocab…")
idx = 0
for ex in ds:
    txt = ex["text"]
    ids = tokenizer.encode(txt)
    # fit as many as we can
    n = min(len(ids), TOTAL_TOKENS - idx)
    fp[idx:idx+n] = np.array(ids[:n], dtype=np.int32)
    idx += n
    if idx >= TOTAL_TOKENS:
        break

fp.flush()

# wrap & save as .pt
tokens = torch.from_numpy(
    np.memmap(MEMMAP_PATH, dtype=np.int32, mode="r", shape=(TOTAL_TOKENS,))
)
torch.save(tokens, PT_PATH)
print(f"Saved {idx:,} tokens → {PT_PATH}")

# clean up memmap file
os.remove(MEMMAP_PATH)
