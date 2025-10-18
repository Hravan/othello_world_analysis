import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from data import get_othello
from data.othello import OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPTConfig, GPTforProbing

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate training data for probes')
    parser.add_argument('--layer', required=True, default=-1, type=int)
    parser.add_argument('--random', dest='random', action='store_true')
    parser.add_argument('--championship', dest='championship', action='store_true')
    parser.add_argument('--exp', default="state", type=str)
    parser.add_argument('--ckpts_path', default="./ckpts", type=str, help="Path to the checkpoints directory")
    parser.add_argument('--output_dir', default="training_data", type=str, help="Path to the output directory")
    parser.add_argument('--device', default=None, type=str, help='Device to run on, e.g. cpu or cuda:0. If not set, auto-detects.')
    return parser.parse_args()

def load_model(args, train_dataset):
    print("Loading GPT model...")
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPTforProbing(mconf, probe_layer=args.layer)
    # determine torch device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load checkpoints with correct map_location
    if args.random:
        model.apply(model._init_weights)
    else:
        ckpt_name = "gpt_championship.ckpt" if args.championship else "gpt_synthetic.ckpt"
        ckpt_path = Path(args.ckpts_path) / ckpt_name
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"Warning: checkpoint {ckpt_path} not found; running with random init")

    model = model.to(device)
    return model

def generate_training_data(model, train_dataset, args):
    print("Generating training data...")
    # pin_memory is only useful when using CUDA; avoid when on CPU
    use_cuda = next(model.parameters()).is_cuda
    loader = DataLoader(train_dataset, shuffle=False, pin_memory=use_cuda, batch_size=1, num_workers=1)
    act_container = []
    property_container = []
    age_container = []

    for x, _ in tqdm(loader, total=len(loader)):
        # x is on CPU by default; send to model device
        model_device = next(model.parameters()).device
        x_dev = x.to(model_device)

        tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999
        a = OthelloBoardState()
        properties = a.get_gt(tbf[:valid_until], "get_" + args.exp)
        act = model(x_dev)[0, ...].detach().cpu()
        act_container.extend([_[0] for _ in act.split(1, dim=0)[:valid_until]])
        property_container.extend(properties)

    for x, y in tqdm(loader, total=len(loader)):
        tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
        valid_until = tbf.index(-100) if -100 in tbf else 999
        a = OthelloBoardState()
        ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
        age_container.extend(ages)

    return act_container, property_container, age_container

def save_training_data(output_dir, act_container, property_container, age_container):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "activations.npy", np.array(act_container))
    np.save(output_dir / "properties.npy", np.array(property_container))
    np.save(output_dir / "ages.npy", np.array(age_container))
    print(f"Training data saved to {output_dir}")

def main():
    setup_logging()
    args = parse_arguments()

    print("Loading Othello dataset...")
    othello = get_othello(data_root="data/othello_championship")
    train_dataset = CharDataset(othello)

    model = load_model(args, train_dataset)
    act_container, property_container, age_container = generate_training_data(model, train_dataset, args)

    output_dir = Path(args.output_dir)
    save_training_data(output_dir, act_container, property_container, age_container)

if __name__ == "__main__":
    main()
