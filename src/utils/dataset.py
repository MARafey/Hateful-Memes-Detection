import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import jsonlines
import cv2

class HatefulMemesDataset(Dataset):
    """
    Dataset class for the Hateful Memes Challenge
    """
    def __init__(self, 
                 data_path, 
                 img_dir, 
                 split='train', 
                 text_model_name='bert-base-uncased', 
                 max_length=128,
                 img_size=224, 
                 transform=None):
        """
        Initialize the dataset
        
        Args:
            data_path (str): Path to the jsonl file containing the data
            img_dir (str): Path to the directory containing the images
            split (str): Dataset split ('train', 'dev', 'test')
            text_model_name (str): Name of the text model to use for tokenization
            max_length (int): Maximum sequence length for tokenization
            img_size (int): Size to resize the images to
            transform (callable): Optional transform to apply to the images
        """
        self.data_path = data_path
        self.img_dir = img_dir
        self.split = split
        self.max_length = max_length
        self.img_size = img_size
        
        # Load data
        self.data = []
        with jsonlines.open(data_path) as reader:
            for item in reader:
                self.data.append(item)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Set up image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process text
        text = item['text']
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load and process image
        img_path = os.path.join(self.img_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get label (if available)
        if 'label' in item:
            label = torch.tensor(item['label'], dtype=torch.long)
        else:
            # For test set without labels
            label = torch.tensor(-1, dtype=torch.long)
        
        # Get ID for tracking
        id = item['id']
        
        return {
            'id': id,
            'text': text,
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'image': image_tensor,
            'label': label
        }

def get_dataloader(data_path, img_dir, split='train', batch_size=32, text_model_name='bert-base-uncased', 
                  max_length=128, img_size=224, transform=None, num_workers=4, shuffle=True):
    """
    Create a dataloader for the specified split
    
    Args:
        data_path (str): Path to the jsonl file
        img_dir (str): Directory containing the images
        split (str): Dataset split ('train', 'dev', 'test')
        batch_size (int): Batch size
        text_model_name (str): Name of the text model to use for tokenization
        max_length (int): Maximum sequence length for tokenization
        img_size (int): Size to resize the images to
        transform (callable): Optional transform to apply to the images
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the dataset
        
    Returns:
        DataLoader: The created dataloader
    """
    dataset = HatefulMemesDataset(
        data_path=data_path,
        img_dir=img_dir,
        split=split,
        text_model_name=text_model_name,
        max_length=max_length,
        img_size=img_size,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 