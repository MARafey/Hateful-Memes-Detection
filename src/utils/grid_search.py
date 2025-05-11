import os
import json
import itertools
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.utils.trainer import Trainer

class GridSearch:
    """
    Grid search for hyperparameter tuning
    """
    def __init__(self, 
                 model_class, 
                 train_loader, 
                 val_loader, 
                 param_grid, 
                 device='cuda', 
                 output_dir='outputs/grid_search',
                 criterion=torch.nn.CrossEntropyLoss(),
                 num_epochs=3,  # Smaller number of epochs for grid search
                 early_stopping_patience=2,
                 experiment_name=None):
        """
        Initialize the grid search
        
        Args:
            model_class: Class of model to grid search
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            param_grid: Dictionary with hyperparameter names as keys and lists of values as values
            device: Device to use ('cuda' or 'cpu')
            output_dir: Directory to save outputs
            criterion: Loss function
            num_epochs: Number of epochs to train each model
            early_stopping_patience: Number of epochs to wait before early stopping
            experiment_name: Name of the experiment (optional)
        """
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_grid = param_grid
        self.device = device
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Generate a timestamp for the experiment
        if experiment_name is None:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
        
        self.output_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate all combinations of hyperparameters
        self.param_combinations = list(self._generate_param_combinations())
        print(f"Grid Search: {len(self.param_combinations)} parameter combinations to try")
        
        # Initialize results tracking
        self.results = []
    
    def _generate_param_combinations(self):
        """
        Generate all combinations of hyperparameters
        
        Returns:
            Generator for parameter combinations
        """
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def run(self):
        """
        Run the grid search
        
        Returns:
            dict: Results of the grid search
        """
        best_val_auc = 0
        best_params = None
        
        # Try all parameter combinations
        for i, params in enumerate(self.param_combinations):
            print(f"\nTrying parameter combination {i+1}/{len(self.param_combinations)}:")
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            # Initialize model with current parameters
            model = self.model_class(**params).to(self.device)
            
            # Get optimizer parameters from the model class
            optimizer_name = params.get('optimizer', 'adam')
            lr = params.get('learning_rate', 0.001)
            weight_decay = params.get('weight_decay', 0)
            
            # Initialize optimizer
            if optimizer_name.lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name.lower() == 'sgd':
                momentum = params.get('momentum', 0.9)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
            # Create model name based on parameters
            model_name = f"model_{i+1}"
            
            # Create output directory for this run
            run_output_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(run_output_dir, exist_ok=True)
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=self.criterion,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                output_dir=run_output_dir,
                model_name=model_name,
                num_epochs=self.num_epochs,
                early_stopping_patience=self.early_stopping_patience
            )
            
            # Train the model
            history = trainer.train()
            
            # Record results
            val_auc = max(history['val_auc'])
            best_epoch = history['val_auc'].index(val_auc) + 1
            
            # Save the history with the parameters
            result = {
                'params': params,
                'val_auc': val_auc,
                'best_epoch': best_epoch,
                'history': history
            }
            
            self.results.append(result)
            
            # Check if this is the best model so far
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_params = params
                print(f"New best model with validation AUC: {val_auc:.4f}")
            
            # Save current results
            self._save_results()
        
        # Print best results
        print("\nGrid Search completed!")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        
        return {'best_params': best_params, 'best_val_auc': best_val_auc, 'all_results': self.results}
    
    def _save_results(self):
        """
        Save grid search results to file
        """
        # Sort results by validation AUC
        sorted_results = sorted(self.results, key=lambda x: x['val_auc'], reverse=True)
        
        # Save to JSON
        with open(os.path.join(self.output_dir, 'grid_search_results.json'), 'w') as f:
            json.dump(sorted_results, f, indent=2)
    
    def get_best_params(self):
        """
        Get the best parameters from the grid search
        
        Returns:
            dict: Best parameters
        """
        if not self.results:
            raise ValueError("Grid search has not been run yet")
        
        # Sort results by validation AUC
        sorted_results = sorted(self.results, key=lambda x: x['val_auc'], reverse=True)
        
        return {
            'best_params': sorted_results[0]['params'],
            'best_val_auc': sorted_results[0]['val_auc']
        } 