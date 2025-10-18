# %%
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mingpt.probe_model import BatteryProbeClassificationTwoLayer

# %%

mid_dim = 128
how_many_history_step_to_use = 99
exp = f"state_tl{mid_dim}"
exp += "_championship"

# %%
probe = BatteryProbeClassificationTwoLayer('cpu', probe_class=3, num_task=64, mid_dim=mid_dim)
load_res = probe.load_state_dict(torch.load(f"./ckpts/battery_othello/{exp}/layer6/checkpoint.ckpt", map_location=torch.device('cpu')))
probe.eval()
# %%
activations = np.load(f"data/probe-data/activations.npy")
gts = np.load('data/probe-data/properties.npy')
# %%
logits = probe(torch.tensor(activations))[0]
# %%
probs = torch.softmax(logits, dim=-1)
preds = torch.argmax(probs, dim=-1)
# %%
preds = preds.numpy()
gts
# %%
accuracies = np.mean(preds == gts, axis=0)
# %%
accuracies.reshape(8, 8)
# %%
fig = sns.heatmap(accuracies.reshape(8, 8), annot=True, fmt=".3f", cmap="Blues")
# %%
fig.figure.savefig(f"board_analysis_{exp}.png")