import os
import sys
import argparse
import torch
import random
import numpy as np
from datetime import datetime

from utils.dataset import get_dataloader
from utils.trainer import Trainer
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
    parser = argparse.ArgumentParser(description='Train a model for hateful memes detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--train_file', type=str, default='train.jsonl', help='Training data file')
    parser.add_argument('--dev_file', type=str, default='dev.jsonl', help='Validation data file')
    parser.add_argument('--test_file', type=str, default='test.jsonl', help='Test data file')
    parser.add_argument('--img_dir', type=str, default='data/img', help='Directory containing images')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='late_fusion', 
                        choices=['text_only', 'image_only', 'late_fusion', 'early_fusion', 'cross_attention'],
                        help='Type of model to train')
    parser.add_argument('--text_model_name', type=str, default='bert-base-uncased', 
                        help='Pretrained text model name')
    parser.add_argument('--img_model_name', type=str, default='resnet50', 
                        help='Pretrained image model name')
    parser.add_argument('--fusion_method', type=str, default='concat', 
                        choices=['concat', 'sum', 'max', 'weighted', 'bilinear'],
                        help='Method for fusion in multimodal models')
    parser.add_argument('--text_pooling_type', type=str, default='cls', 
                        choices=['cls', 'mean', 'max'],
                        help='Pooling type for text features')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--freeze_text_model', action='store_true', help='Freeze text model weights')
    parser.add_argument('--freeze_img_model', action='store_true', help='Freeze image model weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'linear', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for saving')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def get_model(args, device):
    """
    Initialize model based on arguments
    
    Args:
        args (argparse.Namespace): Parsed arguments
        device (str): Device to use
        
    Returns:
        torch.nn.Module: Initialized model
    """
    if args.model_type == 'text_only':
        if args.text_pooling_type != 'cls':
            model = TextWithPoolingModel(
                text_model_name=args.text_model_name,
                pooling_type=args.text_pooling_type,
                dropout=args.dropout,
                freeze_text_model=args.freeze_text_model
            )
        else:
            model = TextOnlyModel(
                text_model_name=args.text_model_name,
                dropout=args.dropout,
                freeze_text_model=args.freeze_text_model
            )
    
    elif args.model_type == 'image_only':
        if 'attention' in args.img_model_name:
            model = ResNetWithAttention(
                img_model_name=args.img_model_name.replace('_attention', ''),
                dropout=args.dropout,
                freeze_img_model=args.freeze_img_model
            )
        else:
            model = ImageOnlyModel(
                img_model_name=args.img_model_name,
                dropout=args.dropout,
                freeze_img_model=args.freeze_img_model
            )
    
    elif args.model_type == 'late_fusion':
        if args.fusion_method == 'attention':
            model = LateFusionWithAttention(
                text_model_name=args.text_model_name,
                img_model_name=args.img_model_name,
                text_pooling_type=args.text_pooling_type,
                dropout=args.dropout,
                freeze_text_model=args.freeze_text_model,
                freeze_img_model=args.freeze_img_model
            )
        else:
            model = LateFusionModel(
                text_model_name=args.text_model_name,
                img_model_name=args.img_model_name,
                fusion_method=args.fusion_method,
                text_pooling_type=args.text_pooling_type,
                dropout=args.dropout,
                freeze_text_model=args.freeze_text_model,
                freeze_img_model=args.freeze_img_model
            )
    
    elif args.model_type == 'early_fusion':
        model = EarlyFusionModel(
            text_model_name=args.text_model_name,
            img_model_name=args.img_model_name,
            fusion_method=args.fusion_method,
            dropout=args.dropout,
            freeze_text_model=args.freeze_text_model,
            freeze_img_model=args.freeze_img_model
        )
    
    elif args.model_type == 'cross_attention':
        model = CrossAttentionFusion(
            text_model_name=args.text_model_name,
            img_model_name=args.img_model_name,
            dropout=args.dropout,
            freeze_text_model=args.freeze_text_model,
            freeze_img_model=args.freeze_img_model
        )
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    return model.to(device)

def get_optimizer(model, args):
    """
    Initialize optimizer based on arguments
    
    Args:
        model (torch.nn.Module): Model
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        torch.optim.Optimizer: Initialized optimizer
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

def get_scheduler(optimizer, args, total_steps):
    """
    Initialize learning rate scheduler based on arguments
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        args (argparse.Namespace): Parsed arguments
        total_steps (int): Total number of training steps
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Initialized scheduler
    """
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps
        )
    elif args.scheduler == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=total_steps
        )
    elif args.scheduler == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2, 
            verbose=True
        )
    else:  # 'none'
        return None

def main():
    """
    Main function to train a model
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
    
    # Create output directory
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Load data
    print("Loading data...")
    train_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, args.train_file),
        img_dir=args.img_dir,
        split='train',
        batch_size=args.batch_size,
        text_model_name=args.text_model_name,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, args.dev_file),
        img_dir=args.img_dir,
        split='dev',
        batch_size=args.batch_size,
        text_model_name=args.text_model_name,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    test_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, args.test_file),
        img_dir=args.img_dir,
        split='test',
        batch_size=args.batch_size,
        text_model_name=args.text_model_name,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Initialize model
    print(f"Initializing {args.model_type} model...")
    model = get_model(args, device)
    
    # Initialize optimizer
    optimizer = get_optimizer(model, args)
    
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.num_epochs
    
    # Initialize scheduler
    scheduler = get_scheduler(optimizer, args, total_steps)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_name=args.model_name,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip=args.grad_clip
    )
    
    # Train model
    print(f"Training {args.model_name} for {args.num_epochs} epochs...")
    history = trainer.train(resume_from=args.resume_from)
    
    print("Training completed!")

if __name__ == '__main__':
    main() 