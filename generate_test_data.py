# set up logging
import logging
import os
from pathlib import Path

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
from mingpt.utils import set_seed

set_seed(42)

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data import get_othello
from data.othello import OthelloBoardState, permit, start_hands
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_model import (BatteryProbeClassification,
                                BatteryProbeClassificationTwoLayer)
from mingpt.probe_trainer import Trainer, TrainerConfig

parser = argparse.ArgumentParser(description='Train classification network')
parser.add_argument('--layer',
                    required=True,
                    type=int)

parser.add_argument('--mode',
                    required=True,
                    choices=['championship', 'random', 'synthetic'],
                    help='Mode for loading the model: championship, random, or synthetic')

parser.add_argument('--exp',
                    default="state", 
                    type=str)

parser.add_argument('--output_path',
                    required=True,
                    type=str,
                    help='Path to save the generated test data')

args, _ = parser.parse_known_args()

othello = get_othello(data_root="data/othello_championship")

train_dataset = CharDataset(othello)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPTforProbing(mconf, probe_layer=args.layer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.mode == 'random':
    model.apply(model._init_weights)
elif args.mode == 'championship':
    load_res = model.load_state_dict(torch.load("./ckpts/gpt_championship.ckpt", map_location=device))
elif args.mode == 'synthetic':
    load_res = model.load_state_dict(torch.load("./ckpts/gpt_synthetic.ckpt", map_location=device))
model = model.to(device)

loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=0)
act_container = []
property_container = []
sequence_length_container = []
for x, y in tqdm(loader, total=len(loader)):
    tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
    valid_until = tbf.index(-100) if -100 in tbf else 999
    a = OthelloBoardState()
    properties = a.get_gt(tbf[:valid_until], "get_" + args.exp)  # [block_size, ]
    act = model(x.to(device))[0, ...].detach().cpu()  # [block_size, f]
    act_container.extend([_[0] for _ in act.split(1, dim=0)[:valid_until]])
    property_container.extend(properties)
    sequence_length_container.extend(range(1, len(tbf[:valid_until]) + 1))
    
age_container = []
for x, y in tqdm(loader, total=len(loader)):
    tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
    valid_until = tbf.index(-100) if -100 in tbf else 999
    a = OthelloBoardState()
    ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
    age_container.extend(ages)

class ProbingDataset(Dataset):
    def __init__(self, act, y, age, seq_length):
        assert len(act) == len(y) == len(age) == len(seq_length)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        self.age = age
        self.seq_length = seq_length
        print(np.sum(np.array(y)==0), np.sum(np.array(y)==1), np.sum(np.array(y)==2))
        
        long_age = []
        for a in age:
            long_age.extend(a)
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        print(counts)
    def __len__(self, ):
        return len(self.y)
    def __getitem__(self, idx):
        return self.act[idx], torch.tensor(self.y[idx]).to(torch.long), torch.tensor(self.age[idx]).to(torch.long), torch.tensor(self.seq_length[idx]).to(torch.long)

probing_dataset = ProbingDataset(act_container, property_container, age_container, sequence_length_container)
train_size = int(0.8 * len(probing_dataset))
test_size = len(probing_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])
sampler = None
train_loader = DataLoader(train_dataset, shuffle=False, sampler=sampler, pin_memory=True, batch_size=128, num_workers=0)
test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=0)

# Ensure the output file path is valid
output_file = Path(args.output_path)
output_file.parent.mkdir(parents=True, exist_ok=True)

# Save the test data to the specified output path
# Collect test data
act_container_test = []
property_container_test = []
sequence_length_container_test = []
age_container_test = []

for x_act, y_prop, y_age, seq_len in tqdm(test_loader, total=len(test_loader)):
    act_container_test.extend(x_act.numpy())
    property_container_test.extend(y_prop.numpy())
    sequence_length_container_test.extend(seq_len.numpy())
    age_container_test.extend(y_age.numpy())

# save all test data into a single numpy file
np.savez(output_file,
         activations=np.array(act_container_test),
         properties=np.array(property_container_test),
         sequence_lengths=np.array(sequence_length_container_test),
         ages=np.array(age_container_test))

print(f"Test data saved to {output_file}")
