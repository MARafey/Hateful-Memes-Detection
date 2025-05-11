import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Trainer class for model training, validation, and testing
    """
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 train_loader, 
                 val_loader, 
                 test_loader=None,
                 device='cuda', 
                 output_dir='outputs',
                 model_name='model',
                 scheduler=None,
                 num_epochs=50, 
                 early_stopping_patience=5,
                 gradient_accumulation_steps=1,
                 grad_clip=None):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            optimizer: The optimizer
            criterion: The loss function
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            test_loader: DataLoader for testing (optional)
            device: Device to use ('cuda' or 'cpu')
            output_dir: Directory to save outputs
            model_name: Name of the model (for saving)
            scheduler: Learning rate scheduler (optional)
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            grad_clip: Gradient clipping value (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.model_name = model_name
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tensorboard'), exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard', model_name))
        
        # Initialize training variables
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [], 
            'train_auc': [], 
            'train_acc': [],
            'val_loss': [], 
            'val_auc': [], 
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average_loss, auc, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, batch in pbar:
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, images)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping if specified
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_description(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {total_loss/(i+1):.4f}")
            
            # Collect predictions and labels for metrics
            preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, preds_binary)
        
        return avg_loss, auc, accuracy
    
    def validate(self, epoch=None):
        """
        Validate the model
        
        Args:
            epoch: Current epoch number (optional)
            
        Returns:
            tuple: (average_loss, auc, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, images)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, preds_binary)
        
        return avg_loss, auc, accuracy
    
    def test(self):
        """
        Test the model on the test set
        
        Returns:
            tuple: (average_loss, auc, accuracy)
        """
        if self.test_loader is None:
            print("No test loader provided")
            return None, None, None
        
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                ids = batch['id']
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, images)
                
                # If test set has labels
                if labels[0].item() != -1:
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                all_preds.extend(preds)
                all_ids.extend(ids.cpu().numpy())
                
                if labels[0].item() != -1:
                    all_labels.extend(labels.cpu().numpy())
        
        # Save predictions to file
        predictions = {
            'id': all_ids,
            'proba': all_preds,
            'label': [1 if p > 0.5 else 0 for p in all_preds]
        }
        
        with open(os.path.join(self.output_dir, f'{self.model_name}_predictions.json'), 'w') as f:
            json.dump(predictions, f)
        
        # If test set has labels, calculate metrics
        if all_labels:
            avg_loss = total_loss / len(self.test_loader)
            auc = roc_auc_score(all_labels, all_preds)
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            accuracy = accuracy_score(all_labels, preds_binary)
            
            return avg_loss, auc, accuracy
        
        return None, None, None
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoints', f'{self.model_name}_checkpoint_epoch_{epoch}.pth'))
        
        # Save as best model if specified
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, f'{self.model_name}_best.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Set weights_only=False to handle NumPy scalars in the checkpoint
        # Note: This is less secure but necessary for backward compatibility with older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']
        
        start_epoch = checkpoint['epoch'] + 1
        
        return start_epoch
    
    def train(self, resume_from=None):
        """
        Train the model
        
        Args:
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            dict: Training history
        """
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming from epoch {start_epoch}")
        
        print(f"Starting training for {self.num_epochs} epochs...")
        
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            # Training step
            start_time = time.time()
            train_loss, train_auc, train_acc = self.train_epoch(epoch)
            train_time = time.time() - start_time
            
            # Validation step
            val_loss, val_auc, val_acc = self.validate(epoch)
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_auc'].append(train_auc)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('AUC/train', train_auc, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('AUC/val', val_auc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} completed in {train_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Check if this is the best model so far
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                print(f"New best model with validation AUC: {val_auc:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Print training summary
        print(f"\nTraining completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch+1}")
        
        # Load best model for testing
        self.load_checkpoint(os.path.join(self.output_dir, f'{self.model_name}_best.pth'))
        
        # Test the model
        if self.test_loader is not None:
            test_loss, test_auc, test_acc = self.test()
            if test_auc is not None:
                print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/test', test_loss, 0)
                self.writer.add_scalar('AUC/test', test_auc, 0)
                self.writer.add_scalar('Accuracy/test', test_acc, 0)
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.history 