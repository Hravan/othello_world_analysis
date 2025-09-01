import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import argparse
from tqdm import tqdm

class SeqRetrievalDataset(Dataset):
    def __init__(self, npz_path, start_token, stop_token, pad_token):
        data = np.load(npz_path, allow_pickle=True)
        self.acts = torch.tensor(data['acts'], dtype=torch.float32)
        # Shift real tokens by +1, prepend start, append stop
        self.seqs = [torch.tensor([start_token] + [t+1 for t in seq] + [stop_token], dtype=torch.long) for seq in data['seqs']]
        self.pad_token = pad_token
    def __len__(self):
        return len(self.acts)
    def __getitem__(self, idx):
        return self.acts[idx], self.seqs[idx]

def collate_fn(batch, pad_token):
    acts, seqs = zip(*batch)
    seq_lens = [len(s) for s in seqs]
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_token)
    return torch.stack(acts), seqs_padded, torch.tensor(seq_lens)

class GRUSeqDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, pad_token):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.embedding = nn.Embedding(num_classes, hidden_dim, padding_idx=pad_token)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.pad_token = pad_token
    def forward(self, x, seqs_in, seq_lens):
        h0 = self.fc(x).unsqueeze(0)
        seqs_emb = self.embedding(seqs_in)
        packed = pack_padded_sequence(seqs_emb, seq_lens, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed, h0)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.out(out)

def train(model, loader, optimizer, criterion, device):
    model.train()
    for acts, seqs, seq_lens in tqdm(loader, desc="Training", leave=False):
        acts, seqs, seq_lens = acts.to(device), seqs.to(device), seq_lens.to(device)
        optimizer.zero_grad()
        logits = model(acts, seqs[:, :-1], seq_lens-1)
        loss = criterion(logits.reshape(-1, logits.size(-1)), seqs[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for acts, seqs, seq_lens in loader:
            acts, seqs, seq_lens = acts.to(device), seqs.to(device), seq_lens.to(device)
            logits = model(acts, seqs[:, :-1], seq_lens-1)
            preds = logits.argmax(-1)
            mask = seqs[:, 1:] != model.pad_token
            correct += ((preds == seqs[:, 1:]) * mask).sum().item()
            total += mask.sum().item()
    print(f"Token accuracy: {correct/total:.4f}")


def generate_sequence(model, act, start_token, stop_token, pad_token, max_len, device):
    model.eval()
    generated = [start_token]
    input_token = torch.tensor([[start_token]], dtype=torch.long, device=device)
    act = act.unsqueeze(0).to(device)  # (1, 512)
    h = model.fc(act).unsqueeze(0)     # (1, 1, H)
    for _ in range(max_len):
        input_oh = torch.nn.functional.one_hot(input_token, num_classes=model.out.out_features).float()
        out, h = model.gru(input_oh, h)
        logits = model.out(out[:, -1])
        next_token = logits.argmax(-1).item()
        if next_token == stop_token or next_token == pad_token:
            break
        generated.append(next_token)
        input_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
    return generated[1:]  # exclude start_token

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GRU sequence decoder on Othello activations.")
    parser.add_argument("train_npz", type=str, help="Path to training .npz file")
    parser.add_argument("test_npz", type=str, help="Path to test .npz file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--real_vocab_size", type=int, required=True, help="Number of real tokens (excluding special tokens)")
    args = parser.parse_args()

    # Token indices
    start_token = 0
    stop_token = args.real_vocab_size + 1
    pad_token = args.real_vocab_size + 2
    num_classes = args.real_vocab_size + 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SeqRetrievalDataset(args.train_npz, start_token, stop_token, pad_token)
    test_dataset = SeqRetrievalDataset(args.test_npz, start_token, stop_token, pad_token)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token)
    )

    model = GRUSeqDecoder(512, args.hidden_dim, num_classes, pad_token).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train(model, train_loader, optimizer, criterion, device)
        evaluate(model, test_loader, criterion, device)

    # Example: generate a sequence for the first test sample
    act = test_dataset.acts[0]
    generated_seq = generate_sequence(model, act, start_token, stop_token, pad_token, max_len=100, device=device)
    print("Generated sequence:", generated_seq)

if __name__ == "__main__":
    main()
