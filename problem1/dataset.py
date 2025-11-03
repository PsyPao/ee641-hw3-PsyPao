"""
Dataset and data loading utilities for addition task.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AdditionDataset(Dataset):
    """
    Dataset for multi-digit addition task.

    Loads pre-generated addition problems from JSON files.
    """

    def __init__(self, data_path, pad_token=0):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file
            pad_token: Token used for padding
        """
        self.pad_token = pad_token

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Determine max lengths for padding
        self.max_input_len = max(len(s['input']) for s in self.data)
        self.max_target_len = max(len(s['target']) for s in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict containing:
                - input: Input sequence tensor
                - target: Target sequence tensor
                - input_len: Original input length
                - target_len: Original target length
        """
        sample = self.data[idx]

        # Convert to tensors
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)

        # Store original lengths
        input_len = len(sample['input'])
        target_len = len(sample['target'])

        # Pad sequences to max length
        input_padded = torch.nn.functional.pad(
            input_seq,
            (0, self.max_input_len - input_len),
            value=self.pad_token
        )
        target_padded = torch.nn.functional.pad(
            target_seq,
            (0, self.max_target_len - target_len),
            value=self.pad_token
        )

        return {
            'input': input_padded,
            'target': target_padded,
            'input_len': input_len,
            'target_len': target_len
        }


def collate_fn(batch):
    """
    Custom collate function for batching.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors
    """
    # Stack all tensors
    inputs = torch.stack([s['input'] for s in batch])
    targets = torch.stack([s['target'] for s in batch])
    input_lens = torch.tensor([s['input_len'] for s in batch])
    target_lens = torch.tensor([s['target_len'] for s in batch])

    return {
        'input': inputs,
        'target': targets,
        'input_len': input_lens,
        'target_len': target_lens
    }


def create_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing train.json, val.json, test.json
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = AdditionDataset(data_dir / 'train.json')
    val_dataset = AdditionDataset(data_dir / 'val.json')
    test_dataset = AdditionDataset(data_dir / 'test.json')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_vocab_size(num_digits=3):
    """
    Calculate vocabulary size for addition task.

    Args:
        num_digits: Number of digits in each operand

    Returns:
        Vocabulary size (digits 0-9 + operator + padding)
    """
    return 12  # 0-9 digits + operator token + padding token