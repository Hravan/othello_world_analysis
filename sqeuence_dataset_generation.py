
"""
Preprocess OthelloGPT hidden states into numpy arrays for sequence retrieval.
Each training sample: (hidden_state at time t, full ordered sequence of tokens [0..t-1]).
Takes a folder of input files and outputs a single .npz file with keys: acts, seqs.
"""


import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Preprocess OthelloGPT hidden states into numpy arrays for sequence retrieval.")
parser.add_argument("input_dir", type=str, help="Directory containing input files (PGN or similar)")
parser.add_argument("output_file", type=str, help="Output .npz file path")
parser.add_argument("--layer", type=int, default=6, help="Probe layer to extract activations from")
parser.add_argument("--ckpt", type=str, default="./ckpts/gpt_championship.ckpt", help="Path to model checkpoint")
args = parser.parse_args()

LAYER = args.layer
CKPT = args.ckpt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Load model + dataset
# ----------------------------
print("Loading dataset from directory...", args.input_dir)
othello = get_othello(data_root=args.input_dir)
dataset = CharDataset(othello)

print("Loading model...")
mconf = GPTConfig(dataset.vocab_size, dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = GPTforProbing(mconf, probe_layer=LAYER)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                    batch_size=1, num_workers=1)


# ----------------------------
# Collect samples
# ----------------------------
acts = []
seqs = []

print("Extracting activations and histories...")
for x, _ in tqdm(loader, total=len(loader)):
    toks = x.tolist()[0]   # sequence of token IDs
    # Forward pass
    act = model(x.to(DEVICE))[0].detach().cpu().numpy()  # [block_size, hidden_dim]

    # For each position t>0, collect (h_t, seq[:t])
    for t in range(1, len(toks)):
        acts.append(act[t])        # hidden state at step t
        seqs.append(toks[:t])      # ordered full history

# ----------------------------
# Save to disk
# ----------------------------
acts_np = np.stack(acts)
seqs_np = np.array(seqs, dtype=object)
np.savez_compressed(args.output_file, acts=acts_np, seqs=seqs_np)

print(f"Saved {len(acts_np)} samples to {args.output_file}")
