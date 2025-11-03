"""
Extrapolation analysis for different positional encoding strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import create_model
from dataset import create_extrapolation_loader, create_dataloaders
from positional_encoding import visualize_positional_encoding


def evaluate_extrapolation(model, data_dir, test_lengths, device, batch_size=32):
    """
    Evaluate model on sequences of different lengths.

    Args:
        model: Trained model
        data_dir: Data directory
        test_lengths: List of sequence lengths to test
        device: Device to run on
        batch_size: Batch size

    Returns:
        Dictionary mapping length to accuracy
    """
    model.eval()
    results = {}

    for length in test_lengths:
        print(f"Testing on length {length}...")

        # Load data for this length
        dataloader = create_extrapolation_loader(
            data_dir, length, batch_size
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Length {length}"):
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                masks = batch['mask'].to(device)

                logits = model(sequences, masks)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        results[length] = accuracy
        print(f"  Accuracy: {accuracy:.2%}")

    return results


def plot_extrapolation_curves(all_results, save_path):
    """
    Plot extrapolation curves for all encoding types.

    Args:
        all_results: Dictionary mapping encoding type to results
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    # Define colors and markers for each encoding type
    styles = {
        'sinusoidal': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'learned': {'color': 'red', 'marker': 's', 'linestyle': '--'},
        'none': {'color': 'gray', 'marker': '^', 'linestyle': ':'}
    }

    for encoding_type, results in all_results.items():
        lengths = sorted(results.keys())
        accuracies = [results[l] for l in lengths]

        style = styles[encoding_type]
        plt.plot(lengths, accuracies,
                label=encoding_type.capitalize(),
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=8,
                linewidth=2)

    # Add training range indicator
    plt.axvspan(8, 16, alpha=0.2, color='green', label='Training Range')

    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Length Extrapolation Performance', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 270)
    plt.ylim(0, 1.05)

    # Add percentage labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved extrapolation curves to {save_path}")


def visualize_learned_positions(model_path, output_dir, max_positions=128):
    """
    Visualize learned positional embeddings.

    Args:
        model_path: Path to trained model with learned encoding
        output_dir: Directory to save visualizations
        max_positions: Number of positions to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with same hyperparameters as training
    model = create_model(
        encoding_type='learned',
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Extract learned positional embeddings
    pos_encoding = model.pos_encoding

    embeddings = pos_encoding.position_embeddings.weight.data.cpu().numpy()
    embeddings = embeddings[:max_positions]

    # Visualize embeddings as heatmap
    plt.figure(figsize=(12, 8))

    im = plt.imshow(embeddings, cmap='RdBu_r', aspect='auto')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Learned Positional Embeddings')
    plt.colorbar(im, label='Embedding Value')

    save_path = output_dir / 'learned_position_embeddings.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved position embeddings visualization to {save_path}")


def compare_position_encodings(output_dir, d_model=128, max_len=128):
    """
    Compare different positional encoding strategies visually.

    Args:
        output_dir: Directory to save visualizations
        d_model: Model dimension
        max_len: Maximum sequence length to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from positional_encoding import (
        SinusoidalPositionalEncoding,
        LearnedPositionalEncoding,
        NoPositionalEncoding
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sinusoidal encoding
    sinusoidal = SinusoidalPositionalEncoding(d_model, max_len)
    sin_encoding = visualize_positional_encoding(sinusoidal, max_len, d_model)

    im1 = axes[0].imshow(sin_encoding[:50, :50], cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Sinusoidal Encoding')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    plt.colorbar(im1, ax=axes[0])

    # Learned encoding (random initialization)
    learned = LearnedPositionalEncoding(d_model, max_len)
    learned_encoding = visualize_positional_encoding(learned, max_len, d_model)

    im2 = axes[1].imshow(learned_encoding[:50, :50], cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Learned Encoding (Init)')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Position')
    plt.colorbar(im2, ax=axes[1])

    # No encoding
    no_encoding = NoPositionalEncoding(d_model, max_len)
    none_encoding = visualize_positional_encoding(no_encoding, max_len, d_model)

    im3 = axes[2].imshow(none_encoding[:50, :50], cmap='RdBu_r', aspect='auto')
    axes[2].set_title('No Encoding')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Position')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Positional Encoding Comparison', fontsize=14)
    plt.tight_layout()

    save_path = output_dir / 'encoding_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved encoding comparison to {save_path}")


def analyze_failure_cases(model, dataloader, device, num_examples=10):
    """
    Analyze failure cases to understand extrapolation failures.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        num_examples: Number of examples to analyze
    """
    model.eval()
    failures = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            # Get predictions
            logits = model(sequences, masks)
            predictions = logits.argmax(dim=1)

            # Find failures
            incorrect = predictions != labels

            for i in range(incorrect.sum().item()):
                if len(failures) >= num_examples:
                    break

                idx = incorrect.nonzero()[i].item()
                seq_len = batch['length'][idx].item()

                failures.append({
                    'sequence': sequences[idx][:seq_len].cpu().numpy(),
                    'true_label': labels[idx].item(),
                    'predicted': predictions[idx].item(),
                    'confidence': torch.softmax(logits[idx], dim=0).max().item(),
                    'length': seq_len
                })

            if len(failures) >= num_examples:
                break

    # Print failure analysis
    print(f"\nAnalyzing {len(failures)} failure cases:")
    for i, failure in enumerate(failures):
        print(f"\nExample {i+1}:")
        print(f"  Length: {failure['length']}")
        print(f"  Sequence (first 10): {failure['sequence'][:10]}...")
        print(f"  True label: {'Sorted' if failure['true_label'] else 'Unsorted'}")
        print(f"  Predicted: {'Sorted' if failure['predicted'] else 'Unsorted'}")
        print(f"  Confidence: {failure['confidence']:.2%}")

    return failures


def main():
    parser = argparse.ArgumentParser(description='Analyze extrapolation performance')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--output-dir', default='results/extrapolation',
                        help='Output directory for analysis')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--test-lengths', type=int, nargs='+',
                        default=[32, 64, 128, 256],
                        help='Sequence lengths to test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test all three encoding types
    encoding_types = ['sinusoidal', 'learned', 'none']
    all_results = {}

    for encoding_type in encoding_types:
        print(f"\n{'='*50}")
        print(f"Testing {encoding_type} encoding")
        print(f"{'='*50}")

        model_path = Path(args.results_dir) / encoding_type / 'best_model.pth'

        if not model_path.exists():
            print(f"Model not found at {model_path}, skipping...")
            continue

        # Load model with same hyperparameters as training
        model = create_model(
            encoding_type=encoding_type,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            dropout=0.2
        ).to(args.device)
        model.load_state_dict(torch.load(model_path))

        # Evaluate extrapolation
        results = evaluate_extrapolation(
            model, args.data_dir, args.test_lengths, args.device, args.batch_size
        )

        all_results[encoding_type] = results

    # Save numerical results
    with open(output_dir / 'extrapolation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot extrapolation curves
    plot_extrapolation_curves(
        all_results,
        output_dir / 'extrapolation_curves.png'
    )

    # Visualize learned positions if available
    learned_model_path = Path(args.results_dir) / 'learned' / 'best_model.pth'
    if learned_model_path.exists():
        visualize_learned_positions(
            learned_model_path,
            output_dir / 'position_viz'
        )

    # Compare encoding strategies
    compare_position_encodings(output_dir / 'encoding_comparison')

    # Analyze failure cases for learned encoding at length 64
    if 'learned' in all_results:
        print("\n" + "="*50)
        print("Analyzing failure cases for learned encoding at length 64")
        print("="*50)

        model = create_model(
            encoding_type='learned',
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            dropout=0.2
        ).to(args.device)
        model.load_state_dict(torch.load(
            Path(args.results_dir) / 'learned' / 'best_model.pth'
        ))

        dataloader = create_extrapolation_loader(args.data_dir, 64, args.batch_size)
        analyze_failure_cases(model, dataloader, args.device)

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
