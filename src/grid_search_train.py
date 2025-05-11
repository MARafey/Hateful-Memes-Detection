import os
import sys
import argparse
import torch
import random
import numpy as np
import json
from datetime import datetime

from utils.dataset import get_dataloader
from utils.grid_search import GridSearch
from models.text_models import TextOnlyModel, TextWithPoolingModel
from models.image_models import ImageOnlyModel, ResNetWithAttention
from models.late_fusion_models import LateFusionModel, LateFusionWithAttention
from models.early_fusion_models import EarlyFusionModel, CrossAttentionFusion

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Grid search for hateful memes detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.jsonl', help='Validation data file')
    parser.add_argument('--img_dir', type=str, default='data/img', help='Directory containing images')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='late_fusion', 
                        choices=['text_only', 'image_only', 'late_fusion', 'early_fusion', 'cross_attention'],
                        help='Type of model to grid search')
    
    # Grid search parameters
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for each grid search run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--grid_config', type=str, default=None, help='JSON file with grid search configuration')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs/grid_search', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def get_model_class(model_type):
    """
    Get model class based on model type
    
    Args:
        model_type (str): Model type
        
    Returns:
        class: Model class
    """
    if model_type == 'text_only':
        return TextWithPoolingModel
    elif model_type == 'image_only':
        return ImageOnlyModel
    elif model_type == 'late_fusion':
        return LateFusionModel
    elif model_type == 'early_fusion':
        return EarlyFusionModel
    elif model_type == 'cross_attention':
        return CrossAttentionFusion
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_default_grid_config(model_type):
    """
    Get default grid search configuration for the specified model type
    
    Args:
        model_type (str): Model type
        
    Returns:
        dict: Grid search configuration
    """
    # Common parameters for all model types
    common_params = {
        'dropout': [0.1, 0.3, 0.5],
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'weight_decay': [0.0, 0.01, 0.1],
        'optimizer': ['adam']
    }
    
    if model_type == 'text_only':
        return {
            **common_params,
            'text_model_name': ['bert-base-uncased', 'distilbert-base-uncased'],
            'pooling_type': ['cls', 'mean', 'max'],
            'freeze_text_model': [False, True]
        }
    
    elif model_type == 'image_only':
        return {
            **common_params,
            'img_model_name': ['resnet50', 'resnet101', 'efficientnet_b0'],
            'freeze_img_model': [False, True]
        }
    
    elif model_type == 'late_fusion':
        return {
            **common_params,
            'text_model_name': ['bert-base-uncased', 'distilbert-base-uncased'],
            'img_model_name': ['resnet50', 'efficientnet_b0'],
            'fusion_method': ['concat', 'sum', 'weighted'],
            'text_pooling_type': ['cls', 'mean'],
            'freeze_text_model': [False, True],
            'freeze_img_model': [False, True]
        }
    
    elif model_type == 'early_fusion':
        return {
            **common_params,
            'text_model_name': ['bert-base-uncased', 'distilbert-base-uncased'],
            'img_model_name': ['resnet50', 'efficientnet_b0'],
            'fusion_method': ['concat', 'sum', 'bilinear'],
            'freeze_text_model': [False, True],
            'freeze_img_model': [False, True]
        }
    
    elif model_type == 'cross_attention':
        return {
            **common_params,
            'text_model_name': ['bert-base-uncased', 'distilbert-base-uncased'],
            'img_model_name': ['resnet50', 'resnet101'],
            'num_attention_heads': [4, 8],
            'freeze_text_model': [False, True],
            'freeze_img_model': [False, True]
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    """
    Main function to run grid search
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"grid_search_{args.model_type}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Get grid search configuration
    if args.grid_config is not None:
        with open(args.grid_config, 'r') as f:
            param_grid = json.load(f)
    else:
        param_grid = get_default_grid_config(args.model_type)
    
    # Save grid search configuration
    with open(os.path.join(output_dir, 'grid_config.json'), 'w') as f:
        json.dump(param_grid, f, indent=2)
    
    # Load data
    print("Loading data...")
    train_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, args.train_file),
        img_dir=args.img_dir,
        split='train',
        batch_size=args.batch_size,
        text_model_name='bert-base-uncased',  # Default for loading, will be overridden in models
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, args.dev_file),
        img_dir=args.img_dir,
        split='dev',
        batch_size=args.batch_size,
        text_model_name='bert-base-uncased',  # Default for loading, will be overridden in models
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Get model class
    model_class = get_model_class(args.model_type)
    
    # Initialize grid search
    grid_search = GridSearch(
        model_class=model_class,
        train_loader=train_loader,
        val_loader=val_loader,
        param_grid=param_grid,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        experiment_name=args.experiment_name
    )
    
    # Run grid search
    print(f"Running grid search for {args.model_type} model...")
    results = grid_search.run()
    
    # Print best parameters
    best_params = results['best_params']
    best_val_auc = results['best_val_auc']
    
    print("\nGrid Search completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Save best parameters to file
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump({'best_params': best_params, 'best_val_auc': best_val_auc}, f, indent=2)
    
    print(f"\nThe grid search results are saved to: {output_dir}")

if __name__ == '__main__':
    main() 