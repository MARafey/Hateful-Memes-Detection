import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
import torch.nn as nn
from torchsummary import summary
import argparse

from utils.dataset import get_dataloader
from models.text_models import TextOnlyModel, TextWithPoolingModel
from models.image_models import ImageOnlyModel, ResNetWithAttention
from models.late_fusion_models import LateFusionModel, LateFusionWithAttention
from models.early_fusion_models import EarlyFusionModel, CrossAttentionFusion

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Create directory if it doesn't exist
    confusion_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(confusion_dir, exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(confusion_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()

def generate_tikz_architecture(model, model_name, output_path):
    """
    Generate TikZ code for visualizing model architecture
    
    Args:
        model: The model
        model_name: Name of the model
        output_path: Path to save the TikZ file
    """
    tikz_code = []
    
    # Add LaTeX preamble
    tikz_code.append(r"""\documentclass{article}
\usepackage{tikz}
\usepackage[utf8]{inputenc}
\usepackage{xcolor}
\usepackage{graphicx}
\usetikzlibrary{shapes.geometric, arrows, positioning, fit, backgrounds, matrix, chains}

\begin{document}
\title{Model Architecture: """ + model_name + r"""}
\author{Hateful Memes Detection}
\maketitle

\begin{center}
\begin{tikzpicture}[
    node distance=1.2cm,
    box/.style={rectangle, draw, minimum width=3cm, minimum height=0.8cm, text centered, font=\small},
    arrow/.style={thick, ->},
    embedding/.style={box, fill=blue!20},
    conv/.style={box, fill=orange!20},
    pool/.style={box, fill=green!20},
    fc/.style={box, fill=purple!20},
    attention/.style={box, fill=red!20},
    fusion/.style={box, fill=yellow!20},
    other/.style={box, fill=gray!20}
]
""")
    
    # Analyze model structure
    layers = []
    layer_types = {}
    
    # Map to store layer style based on class name
    layer_style_map = {
        'Linear': 'fc',
        'Conv2d': 'conv', 
        'MaxPool2d': 'pool',
        'AvgPool2d': 'pool',
        'Embedding': 'embedding',
        'MultiheadAttention': 'attention',
        'TransformerEncoder': 'attention',
        'LSTM': 'attention', 
        'GRU': 'attention',
    }
    
    # Extract layers and their types
    for name, layer in model.named_children():
        layer_type = type(layer).__name__
        layer_style = layer_style_map.get(layer_type, 'other')
        
        if 'attention' in name.lower() or 'transformer' in name.lower():
            layer_style = 'attention'
        elif 'fusion' in name.lower():
            layer_style = 'fusion'
        elif 'pool' in name.lower():
            layer_style = 'pool'
        elif 'conv' in name.lower():
            layer_style = 'conv'
        elif 'embed' in name.lower():
            layer_style = 'embedding'
        elif 'fc' in name.lower() or 'linear' in name.lower() or 'classifier' in name.lower():
            layer_style = 'fc'
        
        layer_info = {
            'name': name,
            'type': layer_type,
            'style': layer_style,
            'shape': 'unknown'
        }
        
        # Try to get input/output dimensions
        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
            layer_info['shape'] = f"{layer.in_features} → {layer.out_features}"
        elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
            layer_info['shape'] = f"{layer.in_channels} → {layer.out_channels}"
            
        layers.append(layer_info)
    
    # Generate TikZ nodes
    for idx, layer in enumerate(layers):
        node_name = f"node{idx}"
        label = f"{layer['name']}\\n{layer['type']}"
        if layer['shape'] != 'unknown':
            label += f"\\n{layer['shape']}"
        
        tikz_code.append(f"    \\node[{layer['style']}] ({node_name}) "
                        f"{'' if idx == 0 else f'[below=of node{idx-1}]'} "
                        f"{{ {label} }};")
    
    # Add input and output nodes
    if layers:
        tikz_code.append(f"    \\node[box, above=of node0] (input) {{Input}};")
        tikz_code.append(f"    \\node[box, below=of node{len(layers)-1}] (output) {{Output}};")
        
        # Connect nodes with arrows
        tikz_code.append(f"    \\draw[arrow] (input) -- (node0);")
        for i in range(len(layers)-1):
            tikz_code.append(f"    \\draw[arrow] (node{i}) -- (node{i+1});")
        tikz_code.append(f"    \\draw[arrow] (node{len(layers)-1}) -- (output);")
    
    # Close TikZ picture and document
    tikz_code.append(r"""
\end{tikzpicture}
\end{center}

% Legend
\begin{center}
\begin{tikzpicture}[
    node distance=0.5cm,
    legend/.style={rectangle, draw, minimum width=2cm, minimum height=0.6cm, text centered, font=\footnotesize},
]
    \node[legend, fill=blue!20] (emb) {Embedding};
    \node[legend, right=of emb, fill=orange!20] (conv) {Convolution};
    \node[legend, right=of conv, fill=green!20] (pool) {Pooling};
    \node[legend, right=of pool, fill=red!20] (att) {Attention};
    \node[legend, below=of emb, fill=purple!20] (fc) {Fully Connected};
    \node[legend, right=of fc, fill=yellow!20] (fusion) {Fusion};
    \node[legend, right=of fusion, fill=gray!20] (other) {Other};
\end{tikzpicture}
\end{center}

\end{document}
""")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(tikz_code))
    
    print(f"TikZ architecture diagram saved to {output_path}")

def summarize_model_architecture(model, model_name, input_shape, output_dir):
    """
    Summarize and save model architecture
    
    Args:
        model: The model
        model_name: Name of the model
        input_shape: Input shape for the model summary
        output_dir: Directory to save the summary
    """
    # Create directories if they don't exist
    architecture_dir = os.path.join(output_dir, 'model_architectures')
    os.makedirs(architecture_dir, exist_ok=True)
    
    tikz_dir = os.path.join(output_dir, 'tikz_architectures')
    os.makedirs(tikz_dir, exist_ok=True)
    
    # Generate TikZ architecture diagram
    tikz_file_path = os.path.join(tikz_dir, f'{model_name}_architecture.tex')
    generate_tikz_architecture(model, model_name, tikz_file_path)
    
    # Redirect stdout to file for text summary
    import sys
    original_stdout = sys.stdout
    with open(os.path.join(architecture_dir, f'{model_name}_architecture.txt'), 'w') as f:
        sys.stdout = f
        
        # Print model architecture
        print(f"Model: {model_name}")
        print(f"Total trainable parameters: {count_parameters(model):,}")
        print("\nModel Architecture:")
        print(model)
        
        # Print layer details if applicable
        try:
            if isinstance(model, nn.Module):
                for name, layer in model.named_children():
                    print(f"\nLayer: {name}")
                    print(f"Type: {type(layer).__name__}")
                    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                        print(f"In features: {layer.in_features}")
                        print(f"Out features: {layer.out_features}")
        except Exception as e:
            print(f"Error printing layer details: {e}")
            
    # Reset stdout
    sys.stdout = original_stdout
    
    print(f"Architecture summary saved to {os.path.join(architecture_dir, f'{model_name}_architecture.txt')}")
    print(f"TikZ diagram saved to {tikz_file_path}")
    
    # Try to compile LaTeX to PDF if pdflatex is available
    try:
        import subprocess
        import shutil
        
        if shutil.which('pdflatex'):
            print(f"Compiling TikZ diagram to PDF...")
            
            # Change to the directory of the .tex file to avoid path issues
            current_dir = os.getcwd()
            os.chdir(tikz_dir)
            
            # Run pdflatex silently
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', f"{model_name}_architecture.tex"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Change back to original directory
            os.chdir(current_dir)
            
            print(f"PDF generated at {os.path.join(tikz_dir, f'{model_name}_architecture.pdf')}")
        else:
            print("pdflatex not found. TikZ files generated but not compiled to PDF.")
    except Exception as e:
        print(f"Error compiling TikZ to PDF: {e}")
        print("You can compile the .tex files manually with pdflatex.")

def evaluate_model(model, model_name, test_loader, device, output_dir):
    """
    Evaluate model and generate metrics including confusion matrix
    
    Args:
        model: The model to evaluate
        model_name: Name of the model
        test_loader: DataLoader for test data
        device: Device to use
        output_dir: Directory to save outputs
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []
    has_ids = False
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Check if 'id' is in batch and handle it properly
            if 'id' in batch:
                has_ids = True
                batch_ids = batch['id']
                if isinstance(batch_ids, torch.Tensor):
                    batch_ids = batch_ids.cpu().numpy().tolist()
                all_ids.extend(batch_ids)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Collect results
            all_probs.extend(probs)
            all_preds.extend(preds)
            
            # Skip samples with -1 labels (unlabeled data)
            valid_indices = (labels != -1).cpu().numpy()
            if np.any(valid_indices):
                all_labels.extend(labels[valid_indices].cpu().numpy())
    
    # Save predictions to file
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    if has_ids:
        predictions = {
            'id': [str(id) for id in all_ids],
            'proba': [float(p) for p in all_probs],
            'label': [int(p > 0.5) for p in all_probs]
        }
    else:
        predictions = {
            'proba': [float(p) for p in all_probs],
            'label': [int(p > 0.5) for p in all_probs]
        }
    
    with open(os.path.join(predictions_dir, f'{model_name}_predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=4)
    
    # Check if we have any valid labels for metrics calculation
    if len(all_labels) < 10:  # Not enough for meaningful metrics
        print(f"Insufficient labeled data for {model_name}, skipping metrics calculation")
        print(f"Generated {len(all_probs)} predictions and saved to file")
        
        # Create a dummy confusion matrix to visualize predictions distribution
        pred_counts = np.bincount(np.array(all_preds), minlength=2)
        
        # Create directory if it doesn't exist
        confusion_dir = os.path.join(output_dir, 'confusion_matrices')
        os.makedirs(confusion_dir, exist_ok=True)
        
        # Plot prediction distribution instead
        plt.figure(figsize=(8, 6))
        plt.bar(['Non-hateful', 'Hateful'], pred_counts)
        plt.title(f'Prediction Distribution - {model_name}')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(confusion_dir, f'{model_name}_pred_distribution.png'), dpi=300)
        plt.close()
        
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1-score': 0.0,
            'predictions_count': len(all_probs),
            'non_hateful_predictions': int(pred_counts[0]),
            'hateful_predictions': int(pred_counts[1])
        }
    
    try:
        # Calculate and plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, ['Non-hateful', 'Hateful'], model_name, output_dir)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=['Non-hateful', 'Hateful'], output_dict=True)
        
        # Save report to file
        reports_dir = os.path.join(output_dir, 'classification_reports')
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, f'{model_name}_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
        }
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {e}")
        print(f"Unique labels: {np.unique(all_labels)}")
        print(f"Unique predictions: {np.unique(all_preds)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1-score': 0.0,
        }

def evaluate_all_models(args):
    """
    Evaluate all models, generate confusion matrices and display architectures
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create report directory
    report_dir = os.path.join(args.output_dir)
    os.makedirs(report_dir, exist_ok=True)
    
    # Use validation set instead of test set for evaluation
    print(f"Using {'validation' if args.use_validation else 'test'} set for evaluation")
    
    # Get appropriate dataloader based on evaluation mode
    if args.use_validation:
        eval_loader = get_dataloader(
            data_path=os.path.join(args.data_dir, "dev.jsonl"),  # dev set has labels
            img_dir=args.img_dir,
            split="dev",
            batch_size=args.batch_size,
            text_model_name=args.text_model,
            num_workers=args.num_workers,
            shuffle=False
        )
    else:
        eval_loader = get_dataloader(
            data_path=os.path.join(args.data_dir, "test.jsonl"),
            img_dir=args.img_dir,
            split="test",
            batch_size=args.batch_size,
            text_model_name=args.text_model,
            num_workers=args.num_workers,
            shuffle=False
        )
    
    # Models are stored in 'outputs' directory, regardless of the output_dir parameter
    models_dir = 'outputs'
    
    # Use the same model naming and configuration as in run_all.py
    model_configs = [
        {
            'name': 'text_only',
            'class': TextOnlyModel,
            'params': {'text_model_name': args.text_model},
            'input_shape': None
        },
        {
            'name': 'text_pooling',
            'class': TextWithPoolingModel,
            'params': {'text_model_name': args.text_model, 'pooling_type': 'mean'},
            'input_shape': None
        },
        {
            'name': 'image_only',
            'class': ImageOnlyModel,
            'params': {'img_model_name': args.img_model},
            'input_shape': None
        },
        {
            'name': 'image_attention',
            'class': ResNetWithAttention,
            'params': {'img_model_name': args.img_model},
            'input_shape': None
        },
        {
            'name': 'late_fusion',
            'class': LateFusionModel,
            'params': {'text_model_name': args.text_model, 'img_model_name': args.img_model},
            'input_shape': None
        },
        {
            'name': 'late_fusion_attention',
            'class': LateFusionWithAttention,
            'params': {'text_model_name': args.text_model, 'img_model_name': args.img_model},
            'input_shape': None
        },
        {
            'name': 'early_fusion',
            'class': EarlyFusionModel,
            'params': {'text_model_name': args.text_model, 'img_model_name': args.img_model},
            'input_shape': None
        },
        {
            'name': 'cross_attention',
            'class': CrossAttentionFusion,
            'params': {'text_model_name': args.text_model, 'img_model_name': args.img_model},
            'input_shape': None
        }
    ]
    
    # Evaluate each model
    results = {}
    
    for config in model_configs:
        model_name = config['name']
        # Path to the best model following the pattern in run_all.py, but using models_dir
        model_path = os.path.join(models_dir, model_name, f"{model_name}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}, skipping evaluation.")
            continue
        
        print(f"===== Evaluating {model_name} =====")
        
        # Initialize model
        model = config['class'](**config['params']).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Summarize model architecture
        summarize_model_architecture(model, model_name, config['input_shape'], report_dir)
        
        # Evaluate model
        metrics = evaluate_model(model, model_name, eval_loader, device, report_dir)
        results[model_name] = metrics
        
        print(f"Metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print()
    
    # Save overall results
    with open(os.path.join(report_dir, "model_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Create comparative plots only if we have metrics
    if results and all(metrics.get('accuracy', 0) > 0 for metrics in results.values()):
        create_comparative_plots(results, report_dir)
    else:
        print("Skipping comparative plots as no valid metrics were calculated")
    
    return results

def create_comparative_plots(results, output_dir):
    """
    Create comparative plots for all models
    
    Args:
        results: Dictionary with model evaluation results
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, 'comparative_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    for metric in metrics:
        values = [results[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, values)
        plt.title(f'Comparative {metric.capitalize()} Across Models')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'comparative_{metric}.png'), dpi=300)
        plt.close()
    
    # Create a single plot with all metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(models))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(index + i * bar_width, values, bar_width, label=metric.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('All Metrics Comparison')
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_comparison.png'), dpi=300)
    plt.close()

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate models and generate confusion matrices')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--img_dir', type=str, default='data', help='Directory containing images')
    
    # Model parameters
    parser.add_argument('--text_model', type=str, default='bert-base-uncased', help='Text model name')
    parser.add_argument('--img_model', type=str, default='resnet50', help='Image model name')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory where to save evaluation reports and visualizations')
    
    # Evaluation parameters
    parser.add_argument('--use_validation', action='store_true', help='Use validation set instead of test set for evaluation')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_all_models(args)

if __name__ == '__main__':
    main() 