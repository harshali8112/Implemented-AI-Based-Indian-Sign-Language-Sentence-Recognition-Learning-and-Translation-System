"""Train the 1D-CNN + Bi-LSTM keypoint classifier."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from keypoint_features import FEATURE_DIM, SEQUENCE_LENGTH

SEED = 42


class KeypointSequenceModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.recurrent = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        outputs, _ = self.recurrent(inputs)
        return self.head(outputs[:, -1])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted(path for path in root.glob("*/*.npy") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No keypoint sequences found below {root}")
    labels = sorted({path.parent.name for path in paths})
    class_to_idx = {label: index for index, label in enumerate(labels)}
    sequences = []
    targets = []
    for path in paths:
        sequence = np.load(path).astype(np.float32)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(f"{path} has shape {sequence.shape}, expected {(SEQUENCE_LENGTH, FEATURE_DIM)}")
        sequences.append(sequence)
        targets.append(class_to_idx[path.parent.name])
    return np.stack(sequences), np.asarray(targets, dtype=np.int64), labels


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(dim=1).cpu() == targets).sum().item()
            count += len(targets)
    return total_loss / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("keypoint_sequences"))
    parser.add_argument("--checkpoint", type=Path, default=Path("isl_keypoint_model.pth"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    seed_everything(SEED)
    sequences, targets, labels = load_dataset(args.data)
    train_indices, test_indices = train_test_split(
        np.arange(len(targets)), test_size=0.15, random_state=SEED, stratify=targets
    )
    train_set = TensorDataset(torch.from_numpy(sequences[train_indices]), torch.from_numpy(targets[train_indices]))
    test_set = TensorDataset(torch.from_numpy(sequences[test_indices]), torch.from_numpy(targets[test_indices]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointSequenceModel(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)

    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs.to(device)), batch_targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch:03d} | test loss {test_loss:.4f} | test accuracy {test_accuracy:.2%}")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": {label: index for index, label in enumerate(labels)},
                    "labels": labels,
                    "input_shape": (SEQUENCE_LENGTH, FEATURE_DIM),
                    "architecture": "conv1d_bilstm",
                },
                args.checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Saved {args.checkpoint} with best test accuracy {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
