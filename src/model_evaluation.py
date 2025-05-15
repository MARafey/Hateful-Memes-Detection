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
    os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices', f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()

def summarize_model_architecture(model, model_name, input_shape, output_dir):
    """
    Summarize and save model architecture
    
    Args:
        model: The model
        model_name: Name of the model
        input_shape: Input shape for the model summary
        output_dir: Directory to save the summary
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'model_architectures'), exist_ok=True)
    
    # Redirect stdout to file
    import sys
    original_stdout = sys.stdout
    with open(os.path.join(output_dir, 'model_architectures', f'{model_name}_architecture.txt'), 'w') as f:
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
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            # Collect results
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, ['Non-hateful', 'Hateful'], model_name, output_dir)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['Non-hateful', 'Hateful'], output_dict=True)
    
    # Save report to file
    os.makedirs(os.path.join(output_dir, 'classification_reports'), exist_ok=True)
    with open(os.path.join(output_dir, 'classification_reports', f'{model_name}_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
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
    
    # Get test dataloader
    test_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, "test.jsonl"),
        img_dir=args.img_dir,
        split="test",
        batch_size=args.batch_size,
        text_model_name=args.text_model,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Define model configurations
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
        model_path = os.path.join(args.output_dir, model_name, f"{model_name}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}, skipping evaluation.")
            continue
        
        print(f"===== Evaluating {model_name} =====")
        
        # Initialize model
        model = config['class'](**config['params']).to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Summarize model architecture
        summarize_model_architecture(model, model_name, config['input_shape'], args.output_dir)
        
        # Evaluate model
        metrics = evaluate_model(model, model_name, test_loader, device, args.output_dir)
        results[model_name] = metrics
        
        print(f"Metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print()
    
    # Save overall results
    with open(os.path.join(args.output_dir, "model_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Create comparative plots
    create_comparative_plots(results, args.output_dir)
    
    return results

def create_comparative_plots(results, output_dir):
    """
    Create comparative plots for all models
    
    Args:
        results: Dictionary with model evaluation results
        output_dir: Directory to save plots
    """
    os.makedirs(os.path.join(output_dir, 'comparative_plots'), exist_ok=True)
    
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
        plt.savefig(os.path.join(output_dir, 'comparative_plots', f'comparative_{metric}.png'), dpi=300)
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
    plt.savefig(os.path.join(output_dir, 'comparative_plots', 'all_metrics_comparison.png'), dpi=300)
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
    parser.add_argument('--img_dir', type=str, default='data/img', help='Directory containing images')
    
    # Model parameters
    parser.add_argument('--text_model', type=str, default='bert-base-uncased', help='Text model name')
    parser.add_argument('--img_model', type=str, default='resnet50', help='Image model name')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_all_models(args)

if __name__ == '__main__':
    main() 