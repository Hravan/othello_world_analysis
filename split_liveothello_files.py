import argparse
from pathlib import Path
import shutil
import random

def main(input_dir, output_dir, test_size=0.2, seed=2137):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if "liveothello" in f.name and f.is_file()]
    random.seed(seed)
    files_shuffled = files[:]
    random.shuffle(files_shuffled)
    split_idx = int(len(files_shuffled) * (1 - test_size))
    train_files = files_shuffled[:split_idx]
    test_files = files_shuffled[split_idx:]

    for f in train_files:
        shutil.copy(f, output_dir / "train" / f.name)
    for f in test_files:
        shutil.copy(f, output_dir / "test" / f.name)

    print(f"Copied {len(train_files)} files to train/, {len(test_files)} files to test/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory to search for files")
    parser.add_argument("output_dir", help="Output directory for train/test split")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
