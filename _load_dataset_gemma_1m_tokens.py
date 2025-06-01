import os
import numpy as np
import torch
from datasets import load_dataset
""" 
In the original repo author pre-tokenized 1 million tokens (Pile + LmSys-chat dataset) into single .pt file

This way, training doesn't call the HuggingFace API every time
"""
#prep cache and streaming dataset
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
os.makedirs(cache_dir, exist_ok=True)
ds_stream = load_dataset(
    "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
    split="train",
    streaming=True,
    cache_dir=cache_dir,
)

# first pass: count total tokens
print("Counting total tokens…")
total_tokens = 0
for ex in ds_stream:
    total_tokens += len(ex["input_ids"])
print(f"→ total_tokens = {total_tokens:,}")

# create memmap
memmap_path = "pile-lmsys-mix-1m-tokenized-gemma-2.dat"
fp = np.memmap(
    memmap_path,
    dtype=np.int32,
    mode="w+",
    shape=(total_tokens,),
)

#second pass: fill it chunk by chunk
print("Writing tokens to memmap…")
idx = 0
ds_stream = load_dataset(
    "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
    split="train",
    streaming=True,
    cache_dir=cache_dir,
)
for ex in ds_stream:
    arr = np.array(ex["input_ids"], dtype=np.int32)
    l = arr.shape[0]
    fp[idx : idx + l] = arr
    idx += l
fp.flush()

# Wrap in torch tensor and save the .pt
print("Converting memmap → torch tensor and saving .pt…")
tokens = torch.from_numpy(
    np.memmap(
        memmap_path,
        dtype=np.int32,
        mode="r",
        shape=(total_tokens,),
    )
)
torch.save(tokens, "pile-lmsys-mix-1m-tokenized-gemma-2.pt")
print("Done!")

# delete memmap file (not needed)
os.remove(memmap_path)
