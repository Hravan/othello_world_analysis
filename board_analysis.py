# %%
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import argparse

from mingpt.probe_model import BatteryProbeClassificationTwoLayer

# %%
parser = argparse.ArgumentParser(description="Board Analysis Script")
parser.add_argument("--mid_dim", type=int, help="Middle dimension size (e.g., 128, 256, etc.)", required=True)
parser.add_argument("--mode", type=str, choices=["championship", "random", "synthetic"], help="Mode of the experiment", required=True)
parser.add_argument("--output_prefix", type=str, help="Prefix for output files (will add _heatmap.png and _by_move.png)", required=True)
parser.add_argument("--data_path", type=str, help="Path to the input data file (.npz)", required=True)
parser.add_argument("--layer", type=int, help="Layer number to load the checkpoint from", required=True)
args = parser.parse_args()

# %%
mid_dim = args.mid_dim
mode = args.mode
output_prefix = args.output_prefix
data_path = args.data_path
layer = args.layer

output_heatmap = f"{output_prefix}_heatmap.png"
output_by_move = f"{output_prefix}_by_move.png"

exp = f"state_tl{mid_dim}"
exp += f"_{mode}"

# %%
probe = BatteryProbeClassificationTwoLayer('cpu', probe_class=3, num_task=64, mid_dim=mid_dim)
load_res = probe.load_state_dict(torch.load(f"./battery_othello/{exp}/layer{layer}/checkpoint.ckpt", map_location=torch.device('cpu')))
probe.eval()
# %%
data = np.load(data_path)
# %%
activations = data['activations']
gts = data['properties']

# %%
logits = probe(torch.tensor(activations))[0]
# %%
probs = torch.softmax(logits, dim=-1)
preds = torch.argmax(probs, dim=-1)
# %%
preds = preds.numpy()
# %%
errors = np.mean(preds != gts, axis=0) * 100  # Convert to percentage
# %%
errors.reshape(8, 8)
# %%
fig = sns.heatmap(errors.reshape(8, 8), annot=True, fmt=".1f", cmap="Reds")
# %%
fig.figure.savefig(output_heatmap)
# %%
seq_length = data['sequence_lengths']
# %%
def plot_tile_error_by_move(errors: np.ndarray, move_counts: np.ndarray):
    """
    Aggregates mean error per move number for each tile and creates an 8x8 grid of line plots.

    Parameters
    ----------
    errors : array-like, shape (n_samples, 64)
        Boolean array where True indicates an error for that tile.
    move_counts : array-like, shape (n_samples,)
        Number of moves corresponding to each row in errors.
    output_path : str
        File path to save the resulting plot.
    """
    errors = np.asarray(errors, dtype=float)
    move_counts = np.asarray(move_counts)

    assert errors.ndim == 2 and errors.shape[1] == 64, "errors must have shape (n_samples, 64)"
    assert move_counts.shape[0] == errors.shape[0], "move_counts length must match number of samples"

    # Convert to DataFrame
    df = pd.DataFrame(errors, columns=[f"tile_{i}" for i in range(64)])
    df["move"] = move_counts

    # Compute mean error per move for each tile
    agg = df.groupby("move", sort=True).mean().reset_index()

    # Melt to long format for seaborn
    melted = agg.melt(id_vars="move", var_name="tile", value_name="error")

    # Plot 8×8 grid
    g = sns.FacetGrid(melted, col="tile", col_wrap=8, sharex=True, sharey=True, height=1.5)
    g.map(sns.lineplot, "move", "error", color="blue", linewidth=0.8)

    g.set_axis_labels("move", "error")
    return g
# %%
error_by_move_plot = plot_tile_error_by_move(preds != gts, seq_length)
# %%
error_by_move_plot.savefig(output_by_move)
