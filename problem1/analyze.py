"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.

    Args:
        model: Trained transformer model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        Dictionary containing attention weights and sample data
    """
    model.eval()

    all_encoder_attentions = []
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            batch_size = inputs.size(0)

            # Hook storage for this batch
            batch_encoder_attentions = []
            batch_decoder_self_attentions = []
            batch_decoder_cross_attentions = []

            # Register hooks to capture attention weights
            def make_hook(attention_list):
                def hook(module, input, output):
                    # output is (attention_output, attention_weights)
                    attention_list.append(output[1].detach().cpu())
                return hook

            # Register hooks on encoder attention layers
            encoder_hooks = []
            for layer in model.encoder_layers:
                hook = layer.self_attention.register_forward_hook(
                    make_hook(batch_encoder_attentions)
                )
                encoder_hooks.append(hook)

            # Register hooks on decoder attention layers
            decoder_self_hooks = []
            decoder_cross_hooks = []
            for layer in model.decoder_layers:
                self_hook = layer.self_attention.register_forward_hook(
                    make_hook(batch_decoder_self_attentions)
                )
                cross_hook = layer.cross_attention.register_forward_hook(
                    make_hook(batch_decoder_cross_attentions)
                )
                decoder_self_hooks.append(self_hook)
                decoder_cross_hooks.append(cross_hook)

            # Forward pass
            decoder_input = targets[:, :-1]
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            _ = model(inputs, decoder_input, tgt_mask=tgt_mask)

            # Remove hooks
            for hook in encoder_hooks + decoder_self_hooks + decoder_cross_hooks:
                hook.remove()

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # Reorganize attention weights by layer
            num_encoder_layers = len(model.encoder_layers)
            num_decoder_layers = len(model.decoder_layers)

            for i in range(num_encoder_layers):
                if i < len(batch_encoder_attentions):
                    if len(all_encoder_attentions) <= i:
                        all_encoder_attentions.append([])
                    all_encoder_attentions[i].append(
                        batch_encoder_attentions[i][:samples_to_take]
                    )

            for i in range(num_decoder_layers):
                if i < len(batch_decoder_self_attentions):
                    if len(all_decoder_self_attentions) <= i:
                        all_decoder_self_attentions.append([])
                    all_decoder_self_attentions[i].append(
                        batch_decoder_self_attentions[i][:samples_to_take]
                    )

                if i < len(batch_decoder_cross_attentions):
                    if len(all_decoder_cross_attentions) <= i:
                        all_decoder_cross_attentions.append([])
                    all_decoder_cross_attentions[i].append(
                        batch_decoder_cross_attentions[i][:samples_to_take]
                    )

            samples_collected += samples_to_take

    # Concatenate attention weights across batches
    encoder_attentions_concat = []
    for layer_attentions in all_encoder_attentions:
        if layer_attentions:
            encoder_attentions_concat.append(torch.cat(layer_attentions, dim=0))

    decoder_self_attentions_concat = []
    for layer_attentions in all_decoder_self_attentions:
        if layer_attentions:
            decoder_self_attentions_concat.append(torch.cat(layer_attentions, dim=0))

    decoder_cross_attentions_concat = []
    for layer_attentions in all_decoder_cross_attentions:
        if layer_attentions:
            decoder_cross_attentions_concat.append(torch.cat(layer_attentions, dim=0))

    return {
        'encoder_attentions': encoder_attentions_concat,
        'decoder_self_attentions': decoder_self_attentions_concat,
        'decoder_cross_attentions': decoder_cross_attentions_concat,
        'inputs': torch.tensor(all_inputs),
        'targets': torch.tensor(all_targets)
    }


def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                               title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, out_len, in_len]
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axes = plt.subplots(
        2, (num_heads + 1) // 2,
        figsize=(5 * ((num_heads + 1) // 2), 8)
    )
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Plot heatmap
        im = ax.imshow(
            attention_weights[head_idx],
            cmap='Blues',
            aspect='auto',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(input_tokens)))
        ax.set_yticks(np.arange(len(output_tokens)))
        ax.set_xticklabels(input_tokens)
        ax.set_yticklabels(output_tokens)

        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_head_specialization(attention_data, output_dir):
    """
    Analyze what each attention head specializes in.

    Args:
        attention_data: Dictionary with attention weights and samples
        output_dir: Directory to save analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze encoder self-attention
    print("Analyzing encoder self-attention patterns...")

    encoder_attentions = attention_data['encoder_attentions']
    inputs = attention_data['inputs']

    head_stats = {'encoder': {}, 'decoder_self': {}, 'decoder_cross': {}}

    # Analyze encoder self-attention patterns
    for layer_idx in range(len(encoder_attentions)):
        layer_stats = {}
        layer_attention = encoder_attentions[layer_idx]  # [batch, heads, seq, seq]

        for head_idx in range(layer_attention.shape[1]):
            head_attention = layer_attention[:, head_idx]  # [batch, seq, seq]

            # Compute average attention to diagonal (same position)
            diag_attention = []
            for i in range(head_attention.shape[1]):
                if i < head_attention.shape[2]:
                    diag_attention.append(head_attention[:, i, i].mean().item())

            # Compute attention entropy
            entropy = -(head_attention * (head_attention + 1e-10).log()).sum(dim=-1).mean().item()

            # Find operator token positions (token 10 is '+')
            operator_positions = (inputs == 10).float()

            # Average attention to operator
            operator_attention = 0
            if operator_positions.sum() > 0:
                for batch_idx in range(head_attention.shape[0]):
                    op_pos = operator_positions[batch_idx].nonzero()
                    if len(op_pos) > 0:
                        operator_attention += head_attention[batch_idx, :, op_pos[0, 0]].mean().item()
                operator_attention /= head_attention.shape[0]

            layer_stats[f'head_{head_idx}'] = {
                'diagonal_attention': np.mean(diag_attention),
                'entropy': entropy,
                'operator_attention': operator_attention
            }

        head_stats['encoder'][f'layer_{layer_idx}'] = layer_stats

    # Save analysis results
    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.

    Test model performance when individual heads are disabled.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Get baseline accuracy
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc, 'encoder': {}, 'decoder': {}}

    # Get number of heads from the first attention layer
    num_heads = model.encoder_layers[0].self_attention.num_heads
    d_model = model.d_model

    # Test each encoder head
    for layer_idx, encoder_layer in enumerate(model.encoder_layers):
        layer_results = {}
        for head_idx in range(num_heads):
            # Save original weights
            original_wo = encoder_layer.self_attention.W_o.weight.data.clone()

            # Zero out this head's contribution
            d_k = d_model // num_heads
            start_idx = head_idx * d_k
            end_idx = (head_idx + 1) * d_k
            encoder_layer.self_attention.W_o.weight.data[:, start_idx:end_idx] = 0

            # Evaluate with this head disabled
            acc = evaluate_model(model, dataloader, device)
            layer_results[f'head_{head_idx}'] = {
                'accuracy': acc,
                'importance': baseline_acc - acc
            }

            # Restore weights
            encoder_layer.self_attention.W_o.weight.data = original_wo

        ablation_results['encoder'][f'layer_{layer_idx}'] = layer_results

    # Test each decoder head
    for layer_idx, decoder_layer in enumerate(model.decoder_layers):
        layer_results = {}

        # Test self-attention heads
        for head_idx in range(num_heads):
            original_wo = decoder_layer.self_attention.W_o.weight.data.clone()

            d_k = d_model // num_heads
            start_idx = head_idx * d_k
            end_idx = (head_idx + 1) * d_k
            decoder_layer.self_attention.W_o.weight.data[:, start_idx:end_idx] = 0

            acc = evaluate_model(model, dataloader, device)
            layer_results[f'self_attn_head_{head_idx}'] = {
                'accuracy': acc,
                'importance': baseline_acc - acc
            }

            decoder_layer.self_attention.W_o.weight.data = original_wo

        # Test cross-attention heads
        for head_idx in range(num_heads):
            original_wo = decoder_layer.cross_attention.W_o.weight.data.clone()

            d_k = d_model // num_heads
            start_idx = head_idx * d_k
            end_idx = (head_idx + 1) * d_k
            decoder_layer.cross_attention.W_o.weight.data[:, start_idx:end_idx] = 0

            acc = evaluate_model(model, dataloader, device)
            layer_results[f'cross_attn_head_{head_idx}'] = {
                'accuracy': acc,
                'importance': baseline_acc - acc
            }

            decoder_layer.cross_attention.W_o.weight.data = original_wo

        ablation_results['decoder'][f'layer_{layer_idx}'] = layer_results

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Create visualization of head importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on

    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            decoder_input = targets[:, :-1]
            decoder_output = targets[:, 1:]

            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)

            outputs = model(inputs, decoder_input, tgt_mask=tgt_mask)
            predictions = outputs.argmax(dim=-1)

            mask = (decoder_output != 0)
            matches = (predictions == decoder_output) | ~mask
            sequence_correct = matches.all(dim=1)

            correct += sequence_correct.sum().item()
            total += inputs.size(0)

    return correct / total if total > 0 else 0


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.

    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save figure
    """
    # Extract performance drops for each head
    baseline = ablation_results['baseline']

    plt.figure(figsize=(15, 8))

    # Collect all head importance scores
    head_labels = []
    importance_scores = []

    # Encoder heads
    for layer_key in sorted(ablation_results.get('encoder', {}).keys()):
        layer_idx = layer_key.split('_')[1]
        for head_key in sorted(ablation_results['encoder'][layer_key].keys()):
            head_idx = head_key.split('_')[1]
            importance = ablation_results['encoder'][layer_key][head_key]['importance']
            head_labels.append(f'Enc-L{layer_idx}-H{head_idx}')
            importance_scores.append(importance)

    # Decoder self-attention heads
    for layer_key in sorted(ablation_results.get('decoder', {}).keys()):
        layer_idx = layer_key.split('_')[1]
        layer_data = ablation_results['decoder'][layer_key]
        for head_key in sorted([k for k in layer_data.keys() if k.startswith('self_attn')]):
            head_idx = head_key.split('_')[-1]
            importance = layer_data[head_key]['importance']
            head_labels.append(f'Dec-Self-L{layer_idx}-H{head_idx}')
            importance_scores.append(importance)

    # Decoder cross-attention heads
    for layer_key in sorted(ablation_results.get('decoder', {}).keys()):
        layer_idx = layer_key.split('_')[1]
        layer_data = ablation_results['decoder'][layer_key]
        for head_key in sorted([k for k in layer_data.keys() if k.startswith('cross_attn')]):
            head_idx = head_key.split('_')[-1]
            importance = layer_data[head_key]['importance']
            head_labels.append(f'Dec-Cross-L{layer_idx}-H{head_idx}')
            importance_scores.append(importance)

    # Create bar plot
    x_pos = np.arange(len(head_labels))
    colors = ['blue' if 'Enc' in label else 'green' if 'Self' in label else 'red'
              for label in head_labels]

    bars = plt.bar(x_pos, importance_scores, color=colors, alpha=0.7)

    # Add value labels on bars
    for bar, score in zip(bars, importance_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1%}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title(f'Head Importance (Baseline: {baseline:.2%})')
    plt.xticks(x_pos, head_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Encoder'),
        Patch(facecolor='green', alpha=0.7, label='Decoder Self-Attention'),
        Patch(facecolor='red', alpha=0.7, label='Decoder Cross-Attention')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    output_dir = Path(output_dir)
    # Examples will be saved in attention_patterns folder

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Take first sample from batch
            input_seq = inputs[0:1]
            target_seq = targets[0]

            # Generate prediction
            # TODO: Use model.generate() to get prediction
            prediction = model.generate(input_seq, max_len=target_seq.size(0))

            # Convert to strings for visualization
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq.cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")

            # Extract attention for this example
            batch_encoder_attentions = []
            batch_decoder_cross_attentions = []

            def make_hook(attention_list):
                def hook(module, input, output):
                    attention_list.append(output[1].detach().cpu())
                return hook

            # Register hooks
            encoder_hooks = []
            for layer in model.encoder_layers:
                hook = layer.self_attention.register_forward_hook(
                    make_hook(batch_encoder_attentions)
                )
                encoder_hooks.append(hook)

            decoder_cross_hooks = []
            for layer in model.decoder_layers:
                cross_hook = layer.cross_attention.register_forward_hook(
                    make_hook(batch_decoder_cross_attentions)
                )
                decoder_cross_hooks.append(cross_hook)

            # Forward pass for this example
            with torch.no_grad():
                decoder_input = targets[0:1, :-1]
                tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
                decoder_output = targets[0:1, 1:]
                _ = model(inputs[0:1], decoder_input, tgt_mask=tgt_mask)

            # Remove hooks
            for hook in encoder_hooks + decoder_cross_hooks:
                hook.remove()

            # Visualize cross-attention for this example (shows alignment)
            if batch_decoder_cross_attentions:
                # Use last decoder layer's cross-attention
                cross_attention = batch_decoder_cross_attentions[-1][0]  # [num_heads, tgt_len, src_len]

                # Average across heads for clearer visualization
                avg_cross_attention = cross_attention.mean(dim=0).numpy()

                plt.figure(figsize=(10, 6))
                im = plt.imshow(avg_cross_attention, cmap='Blues', aspect='auto', vmin=0, vmax=1)
                plt.colorbar(im)

                # Set labels
                input_tokens = [str(t) if t < 10 else '+' if t == 10 else 'PAD' for t in inputs[0].cpu().numpy()]
                output_tokens = [str(t) if t < 10 else 'PAD' for t in decoder_output[0].cpu().numpy()]

                plt.xticks(range(len(input_tokens)), input_tokens)
                plt.yticks(range(len(output_tokens)), output_tokens)
                plt.xlabel('Input Position')
                plt.ylabel('Output Position')
                plt.title(f'Example {batch_idx + 1}: Cross-Attention (Alignment)')

                save_path = output_dir / 'attention_patterns' / f'prediction_example_{batch_idx}_cross_attention.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', default='results/best_model.pth', help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    # Visualize attention patterns for some examples
    print("Visualizing attention patterns...")

    # Get multiple examples for visualization
    num_viz_examples = min(3, len(attention_data['inputs']))  # Visualize up to 3 examples

    if len(attention_data['encoder_attentions']) > 0 and num_viz_examples > 0:

        # For each example
        for example_idx in range(num_viz_examples):
            # Convert tokens to strings for visualization
            input_example = attention_data['inputs'][example_idx].numpy()
            input_tokens = [str(t) if t < 10 else '+' if t == 10 else 'PAD' for t in input_example]

            if example_idx < len(attention_data['targets']):
                target_example = attention_data['targets'][example_idx].numpy()
                target_tokens = [str(t) if t < 10 else 'PAD' for t in target_example[:-1]]

            # Visualize encoder attention patterns - each head separately
            for layer_idx, layer_attention in enumerate(attention_data['encoder_attentions']):
                if example_idx < layer_attention.shape[0]:
                    attention_example = layer_attention[example_idx]  # [num_heads, seq_len, seq_len]

                    # Save full layer visualization
                    visualize_attention_pattern(
                        attention_example.numpy(),
                        input_tokens,
                        input_tokens,
                        title=f"Example {example_idx+1} - Encoder Layer {layer_idx + 1} (All Heads)",
                        save_path=output_dir / 'attention_patterns' / f'example{example_idx}_encoder_L{layer_idx}_all_heads.png'
                    )

                    # Also save individual head visualizations for the first example
                    if example_idx == 0:
                        for head_idx in range(attention_example.shape[0]):
                            plt.figure(figsize=(8, 6))
                            im = plt.imshow(
                                attention_example[head_idx].numpy(),
                                cmap='Blues',
                                aspect='auto',
                                vmin=0,
                                vmax=1,
                                interpolation='nearest'
                            )
                            plt.colorbar(im)

                            plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
                            plt.yticks(range(len(input_tokens)), input_tokens)
                            plt.xlabel('Key Position')
                            plt.ylabel('Query Position')
                            plt.title(f'Encoder Layer {layer_idx + 1}, Head {head_idx + 1}')
                            plt.tight_layout()

                            save_path = output_dir / 'attention_patterns' / f'encoder_L{layer_idx}_H{head_idx}.png'
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()

            # Visualize decoder self-attention patterns
            for layer_idx, layer_attention in enumerate(attention_data['decoder_self_attentions']):
                if example_idx < layer_attention.shape[0]:
                    attention_example = layer_attention[example_idx]

                    # Save full layer visualization
                    visualize_attention_pattern(
                        attention_example.numpy(),
                        target_tokens,
                        target_tokens,
                        title=f"Example {example_idx+1} - Decoder Layer {layer_idx + 1} Self-Attention (All Heads)",
                        save_path=output_dir / 'attention_patterns' / f'example{example_idx}_decoder_self_L{layer_idx}_all_heads.png'
                    )

                    # Individual heads for first example
                    if example_idx == 0:
                        for head_idx in range(attention_example.shape[0]):
                            plt.figure(figsize=(8, 6))
                            im = plt.imshow(
                                attention_example[head_idx].numpy(),
                                cmap='Blues',
                                aspect='auto',
                                vmin=0,
                                vmax=1,
                                interpolation='nearest'
                            )
                            plt.colorbar(im)

                            plt.xticks(range(len(target_tokens)), target_tokens, rotation=45)
                            plt.yticks(range(len(target_tokens)), target_tokens)
                            plt.xlabel('Key Position')
                            plt.ylabel('Query Position')
                            plt.title(f'Decoder Self-Attention Layer {layer_idx + 1}, Head {head_idx + 1}')
                            plt.tight_layout()

                            save_path = output_dir / 'attention_patterns' / f'decoder_self_L{layer_idx}_H{head_idx}.png'
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()

            # Visualize decoder cross-attention patterns
            for layer_idx, layer_attention in enumerate(attention_data['decoder_cross_attentions']):
                if example_idx < layer_attention.shape[0]:
                    attention_example = layer_attention[example_idx]

                    # Save full layer visualization
                    visualize_attention_pattern(
                        attention_example.numpy(),
                        input_tokens,
                        target_tokens,
                        title=f"Example {example_idx+1} - Decoder Layer {layer_idx + 1} Cross-Attention (All Heads)",
                        save_path=output_dir / 'attention_patterns' / f'example{example_idx}_decoder_cross_L{layer_idx}_all_heads.png'
                    )

                    # Individual heads for first example
                    if example_idx == 0:
                        for head_idx in range(attention_example.shape[0]):
                            plt.figure(figsize=(8, 6))
                            im = plt.imshow(
                                attention_example[head_idx].numpy(),
                                cmap='Blues',
                                aspect='auto',
                                vmin=0,
                                vmax=1,
                                interpolation='nearest'
                            )
                            plt.colorbar(im)

                            plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
                            plt.yticks(range(len(target_tokens)), target_tokens)
                            plt.xlabel('Input Position')
                            plt.ylabel('Output Position')
                            plt.title(f'Decoder Cross-Attention Layer {layer_idx + 1}, Head {head_idx + 1}')
                            plt.tight_layout()

                            save_path = output_dir / 'attention_patterns' / f'decoder_cross_L{layer_idx}_H{head_idx}.png'
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()

    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data, output_dir / 'head_analysis'
    )

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()