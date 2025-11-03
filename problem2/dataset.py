"""
Dataset utilities for sorting detection task.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SortingDataset(Dataset):
    """
    Dataset for sorting detection task.
    """

    def __init__(self, data_path, max_len=None, pad_value=100):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON data file
            max_len: Maximum sequence length (for padding)
            pad_value: Value used for padding (should be outside vocab)
        """
        self.pad_value = pad_value

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Determine max length
        if max_len is None:
            self.max_len = max(sample['length'] for sample in self.data)
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - sequence: Padded sequence tensor
                - label: Binary label (0/1)
                - length: Original sequence length
                - mask: Attention mask (1 for real, 0 for padding)
        """
        sample = self.data[idx]

        sequence = sample['sequence']
        label = sample['is_sorted']
        length = sample['length']

        # Convert to tensor
        seq_tensor = torch.tensor(sequence, dtype=torch.long)

        # Pad if necessary
        if len(sequence) < self.max_len:
            padding = torch.full(
                (self.max_len - len(sequence),),
                self.pad_value,
                dtype=torch.long
            )
            seq_tensor = torch.cat([seq_tensor, padding])
        elif len(sequence) > self.max_len:
            # Truncate if sequence is too long
            seq_tensor = seq_tensor[:self.max_len]
            length = self.max_len

        # Create mask
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[:length] = 1.0

        return {
            'sequence': seq_tensor,
            'label': label,
            'length': length,
            'mask': mask
        }


class ExtrapolationDataset(Dataset):
    """
    Dataset for testing extrapolation to specific lengths.
    """

    def __init__(self, data_dir, target_length, pad_value=100):
        """
        Initialize extrapolation dataset.

        Args:
            data_dir: Directory containing extrapolation test files
            target_length: Target sequence length (32, 64, 128, 256)
            pad_value: Padding value
        """
        self.pad_value = pad_value
        self.target_length = target_length

        # Load specific length dataset
        data_path = Path(data_dir) / f'test_len_{target_length}.json'
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get sample without padding (all same length)."""
        sample = self.data[idx]

        sequence = torch.tensor(sample['sequence'], dtype=torch.long)
        label = sample['is_sorted']

        # No padding needed - all sequences are target_length
        mask = torch.ones(self.target_length, dtype=torch.float32)

        return {
            'sequence': sequence,
            'label': label,
            'length': self.target_length,
            'mask': mask
        }


def collate_fn(batch):
    """
    Custom collate function for batching.

    Args:
        batch: List of samples

    Returns:
        Batched tensors
    """
    sequences = torch.stack([s['sequence'] for s in batch])
    labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)
    lengths = torch.tensor([s['length'] for s in batch], dtype=torch.long)
    masks = torch.stack([s['mask'] for s in batch])

    return {
        'sequence': sequences,
        'label': labels,
        'length': lengths,
        'mask': masks
    }


def create_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory containing data files
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = SortingDataset(data_dir / 'train.json')
    val_dataset = SortingDataset(data_dir / 'val.json')
    test_dataset = SortingDataset(data_dir / 'test.json')

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


def create_extrapolation_loader(data_dir, target_length, batch_size=32):
    """
    Create dataloader for extrapolation testing.

    Args:
        data_dir: Directory containing extrapolation data
        target_length: Target sequence length
        batch_size: Batch size

    Returns:
        DataLoader for extrapolation testing
    """
    dataset = ExtrapolationDataset(
        Path(data_dir) / 'extrapolation',
        target_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )