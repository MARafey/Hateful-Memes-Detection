import os
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from tqdm import tqdm

from src.utils.dataset import get_dataloader, HatefulMemesDataset
from src.utils.trainer import Trainer
from src.models.text_models import TextOnlyModel, TextWithPoolingModel
from src.models.image_models import ImageOnlyModel, ResNetWithAttention
from src.models.late_fusion_models import LateFusionModel, LateFusionWithAttention
from src.models.early_fusion_models import EarlyFusionModel, CrossAttentionFusion

def set_seed(seed):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run all models for Hateful Memes")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the data")
    parser.add_argument("--img_dir", type=str, default="data/img", help="Directory containing the images")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model arguments
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", help="Text model name")
    parser.add_argument("--img_model", type=str, default="resnet50", help="Image model name")
    
    # Run modes
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the models")
    parser.add_argument("--interface", action="store_true", help="Launch the interface")
    parser.add_argument("--skip_models", type=str, nargs="+", default=[], 
                        help="Models to skip (text_only, text_pooling, image_only, image_attention, late_fusion, late_fusion_attention, early_fusion, cross_attention)")
    
    return parser.parse_args()

def get_models_and_names(args, device):
    """
    Get all model instances and their names
    """
    models = []
    model_names = []
    
    # Text-only models
    if "text_only" not in args.skip_models:
        text_only = TextOnlyModel(
            text_model_name=args.text_model,
            dropout=0.1,
            freeze_text_model=False
        ).to(device)
        models.append(text_only)
        model_names.append("text_only")
    
    if "text_pooling" not in args.skip_models:
        text_pooling = TextWithPoolingModel(
            text_model_name=args.text_model,
            pooling_type='mean',
            dropout=0.1,
            freeze_text_model=False
        ).to(device)
        models.append(text_pooling)
        model_names.append("text_pooling")
    
    # Image-only models
    if "image_only" not in args.skip_models:
        image_only = ImageOnlyModel(
            img_model_name=args.img_model,
            pretrained=True,
            dropout=0.1,
            freeze_img_model=False
        ).to(device)
        models.append(image_only)
        model_names.append("image_only")
    
    if "image_attention" not in args.skip_models:
        image_attention = ResNetWithAttention(
            img_model_name=args.img_model,
            pretrained=True,
            dropout=0.1,
            freeze_img_model=False
        ).to(device)
        models.append(image_attention)
        model_names.append("image_attention")
    
    # Late fusion models
    if "late_fusion" not in args.skip_models:
        late_fusion = LateFusionModel(
            text_model_name=args.text_model,
            img_model_name=args.img_model,
            fusion_method='concat',
            text_pooling_type='cls',
            dropout=0.1,
            freeze_text_model=False,
            freeze_img_model=False
        ).to(device)
        models.append(late_fusion)
        model_names.append("late_fusion")
    
    if "late_fusion_attention" not in args.skip_models:
        late_fusion_attention = LateFusionWithAttention(
            text_model_name=args.text_model,
            img_model_name=args.img_model,
            text_pooling_type='cls',
            dropout=0.1,
            freeze_text_model=False,
            freeze_img_model=False
        ).to(device)
        models.append(late_fusion_attention)
        model_names.append("late_fusion_attention")
    
    # Early fusion models
    if "early_fusion" not in args.skip_models:
        early_fusion = EarlyFusionModel(
            text_model_name=args.text_model,
            img_model_name=args.img_model,
            fusion_method='concat',
            dropout=0.1,
            freeze_text_model=False,
            freeze_img_model=False
        ).to(device)
        models.append(early_fusion)
        model_names.append("early_fusion")
    
    if "cross_attention" not in args.skip_models:
        cross_attention = CrossAttentionFusion(
            text_model_name=args.text_model,
            img_model_name=args.img_model,
            dropout=0.1,
            freeze_text_model=False,
            freeze_img_model=False
        ).to(device)
        models.append(cross_attention)
        model_names.append("cross_attention")
    
    return models, model_names

def train_all_models(args, device):
    """
    Train all models
    """
    # Get dataloaders
    train_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, "train.jsonl"),
        img_dir=args.img_dir,
        split="train",
        batch_size=args.batch_size,
        text_model_name=args.text_model,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        data_path=os.path.join(args.data_dir, "dev.jsonl"),
        img_dir=args.img_dir,
        split="dev",
        batch_size=args.batch_size,
        text_model_name=args.text_model,
        num_workers=args.num_workers
    )
    
    # Get all models
    models, model_names = get_models_and_names(args, device)
    
    # Train each model
    results = {}
    for model, model_name in zip(models, model_names):
        print(f"===== Training {model_name} =====")
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # Initialize criterion
        criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize scheduler
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=os.path.join(args.output_dir, model_name),
            model_name=model_name,
            scheduler=scheduler,
            num_epochs=args.epochs,
            early_stopping_patience=3
        )
        
        # Train the model
        history = trainer.train()
        
        # Save the results
        results[model_name] = {
            'best_val_auc': trainer.best_val_auc,
            'best_epoch': trainer.best_epoch,
            'history': history
        }
    
    # Save overall results
    with open(os.path.join(args.output_dir, "all_models_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def evaluate_all_models(args, device):
    """
    Evaluate all models on the test set
    """
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
    
    # Get all model names
    _, model_names = get_models_and_names(args, device)
    
    # Evaluate each model
    results = {}
    for model_name in model_names:
        print(f"===== Evaluating {model_name} =====")
        
        # Load the best model
        model_path = os.path.join(args.output_dir, model_name, f"{model_name}_best.pth")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}, skipping evaluation.")
            continue
        
        # Initialize model based on name
        if model_name == "text_only":
            model = TextOnlyModel(text_model_name=args.text_model).to(device)
        elif model_name == "text_pooling":
            model = TextWithPoolingModel(text_model_name=args.text_model, pooling_type='mean').to(device)
        elif model_name == "image_only":
            model = ImageOnlyModel(img_model_name=args.img_model).to(device)
        elif model_name == "image_attention":
            model = ResNetWithAttention(img_model_name=args.img_model).to(device)
        elif model_name == "late_fusion":
            model = LateFusionModel(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "late_fusion_attention":
            model = LateFusionWithAttention(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "early_fusion":
            model = EarlyFusionModel(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "cross_attention":
            model = CrossAttentionFusion(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        else:
            print(f"Unknown model type: {model_name}")
            continue
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize criterion
        criterion = torch.nn.CrossEntropyLoss()
        
        # Evaluate the model
        total_loss = 0
        all_preds = []
        all_labels = []
        all_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
                # Move data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                ids = batch['id']
                
                # Forward pass
                outputs = model(input_ids, attention_mask, images)
                
                # If test set has labels
                if labels[0].item() != -1:
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                all_preds.extend(preds)
                all_ids.extend(ids)
                
                if labels[0].item() != -1:
                    all_labels.extend(labels.cpu().numpy())
        
        # Save predictions to file
        predictions = {
            'id': all_ids,
            'proba': [float(p) for p in all_preds],
            'label': [1 if p > 0.5 else 0 for p in all_preds]
        }
        
        with open(os.path.join(args.output_dir, f'{model_name}_test_predictions.json'), 'w') as f:
            json.dump(predictions, f)
        
        # If test set has labels, calculate metrics
        if all_labels:
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            avg_loss = total_loss / len(test_loader)
            auc = roc_auc_score(all_labels, all_preds)
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            accuracy = accuracy_score(all_labels, preds_binary)
            
            results[model_name] = {
                'test_loss': avg_loss,
                'test_auc': auc,
                'test_accuracy': accuracy
            }
            
            print(f"Test Loss: {avg_loss:.4f} | Test AUC: {auc:.4f} | Test Acc: {accuracy:.4f}")
    
    # Save overall results
    with open(os.path.join(args.output_dir, "all_models_test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_interface(args, device):
    """
    Create an interface for inference
    """
    # Load the tokenizer and transforms
    from transformers import AutoTokenizer
    import torchvision.transforms as transforms
    
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all model names
    _, model_names = get_models_and_names(args, device)
    
    # Load models
    loaded_models = {}
    for model_name in model_names:
        model_path = os.path.join(args.output_dir, model_name, f"{model_name}_best.pth")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}, skipping.")
            continue
        
        # Initialize model based on name
        if model_name == "text_only":
            model = TextOnlyModel(text_model_name=args.text_model).to(device)
        elif model_name == "text_pooling":
            model = TextWithPoolingModel(text_model_name=args.text_model, pooling_type='mean').to(device)
        elif model_name == "image_only":
            model = ImageOnlyModel(img_model_name=args.img_model).to(device)
        elif model_name == "image_attention":
            model = ResNetWithAttention(img_model_name=args.img_model).to(device)
        elif model_name == "late_fusion":
            model = LateFusionModel(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "late_fusion_attention":
            model = LateFusionWithAttention(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "early_fusion":
            model = EarlyFusionModel(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        elif model_name == "cross_attention":
            model = CrossAttentionFusion(text_model_name=args.text_model, img_model_name=args.img_model).to(device)
        else:
            print(f"Unknown model type: {model_name}")
            continue
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        loaded_models[model_name] = model
    
    def predict(image, text, model_type):
        """
        Make a prediction using the selected model
        """
        if model_type not in loaded_models:
            return {"Hateful": 0.0, "Not Hateful": 1.0}, "Model not loaded"
        
        model = loaded_models[model_type]
        
        # Process text
        text_encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)
        
        # Process image
        if image is None:
            return {"Hateful": 0.0, "Not Hateful": 1.0}, "No image provided"
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Create result dictionary
        result = {
            "Hateful": float(probs[1]),
            "Not Hateful": float(probs[0])
        }
        
        # Determine if the meme is hateful
        is_hateful = "The meme is hateful" if probs[1] > 0.5 else "The meme is not hateful"
        
        return result, is_hateful
    
    # Create the interface
    with gr.Blocks(title="Hateful Memes Detector") as demo:
        gr.Markdown("# Hateful Memes Detector")
        gr.Markdown("Upload an image and enter text to check if the meme is hateful")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Meme Image")
                text_input = gr.Textbox(label="Meme Text")
                model_dropdown = gr.Dropdown(
                    choices=list(loaded_models.keys()),
                    label="Model",
                    value=list(loaded_models.keys())[0] if loaded_models else None
                )
                submit_button = gr.Button("Detect")
            
            with gr.Column():
                label_output = gr.Label(label="Prediction")
                text_output = gr.Textbox(label="Result")
        
        submit_button.click(
            fn=predict,
            inputs=[image_input, text_input, model_dropdown],
            outputs=[label_output, text_output]
        )
    
    # Launch the interface
    demo.launch(share=True)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train models if requested
    if args.train:
        train_all_models(args, device)
    
    # Evaluate models if requested
    if args.evaluate:
        evaluate_all_models(args, device)
    
    # Create interface if requested
    if args.interface:
        create_interface(args, device)
    
    # If no mode is specified, train and evaluate
    if not (args.train or args.evaluate or args.interface):
        print("No mode specified, running train and evaluate by default")
        train_all_models(args, device)
        evaluate_all_models(args, device)

if __name__ == "__main__":
    main() 