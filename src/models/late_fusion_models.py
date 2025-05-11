import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.text_models import TextOnlyModel, TextWithPoolingModel
from src.models.image_models import ImageOnlyModel, ResNetWithAttention

class LateFusionModel(nn.Module):
    """
    Late fusion model combining predictions from separate text and image models
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased',
                 img_model_name='resnet50',
                 fusion_method='concat',
                 text_pooling_type='cls',
                 dropout=0.1,
                 freeze_text_model=False,
                 freeze_img_model=False,
                 text_weight=0.5,
                 img_weight=0.5):
        """
        Initialize the late fusion model
        
        Args:
            text_model_name (str): Name of the pretrained text model
            img_model_name (str): Name of the pretrained image model
            fusion_method (str): Method for fusion ('concat', 'sum', 'max', 'weighted')
            text_pooling_type (str): Type of pooling for text features ('cls', 'mean', 'max')
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
            freeze_img_model (bool): Whether to freeze the image model weights
            text_weight (float): Weight for text predictions in weighted fusion
            img_weight (float): Weight for image predictions in weighted fusion
        """
        super(LateFusionModel, self).__init__()
        
        # Initialize text model
        self.text_model = TextWithPoolingModel(
            text_model_name=text_model_name,
            pooling_type=text_pooling_type,
            dropout=dropout,
            freeze_text_model=freeze_text_model
        )
        
        # Initialize image model
        self.img_model = ImageOnlyModel(
            img_model_name=img_model_name,
            dropout=dropout,
            freeze_img_model=freeze_img_model
        )
        
        # Fusion parameters
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.img_weight = img_weight
        
        # If using concatenation fusion, need an extra layer for final prediction
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(4, 2)  # 2 classes from each model
            self._init_weights(self.fusion_layer)
    
    def _init_weights(self, module):
        """
        Initialize the weights
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get predictions from text model
        text_logits = self.text_model(input_ids, attention_mask)
        
        # Get predictions from image model
        img_logits = self.img_model(images=images)
        
        # Fusion of predictions
        if self.fusion_method == 'concat':
            # Concatenate logits and pass through fusion layer
            concat_logits = torch.cat([text_logits, img_logits], dim=1)
            logits = self.fusion_layer(concat_logits)
            
        elif self.fusion_method == 'sum':
            # Sum the logits
            logits = text_logits + img_logits
            
        elif self.fusion_method == 'max':
            # Take the max of logits
            logits = torch.max(text_logits, img_logits)
            
        elif self.fusion_method == 'weighted':
            # Weighted sum of logits
            logits = self.text_weight * text_logits + self.img_weight * img_logits
            
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        return logits


class LateFusionWithAttention(nn.Module):
    """
    Late fusion model with attention mechanism for combining predictions
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased',
                 img_model_name='resnet50',
                 text_pooling_type='cls',
                 dropout=0.1,
                 freeze_text_model=False,
                 freeze_img_model=False):
        """
        Initialize the late fusion model with attention
        
        Args:
            text_model_name (str): Name of the pretrained text model
            img_model_name (str): Name of the pretrained image model
            text_pooling_type (str): Type of pooling for text features ('cls', 'mean', 'max')
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
            freeze_img_model (bool): Whether to freeze the image model weights
        """
        super(LateFusionWithAttention, self).__init__()
        
        # Initialize text model
        self.text_model = TextWithPoolingModel(
            text_model_name=text_model_name,
            pooling_type=text_pooling_type,
            dropout=dropout,
            freeze_text_model=freeze_text_model
        )
        
        # Initialize image model
        self.img_model = ImageOnlyModel(
            img_model_name=img_model_name,
            dropout=dropout,
            freeze_img_model=freeze_img_model
        )
        
        # Attention weights for fusion
        self.attention_weights = nn.Parameter(torch.ones(2, 2))  # Learnable weights for text and image predictions
        
        # Softmax for normalizing attention weights
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get predictions from text model
        text_logits = self.text_model(input_ids, attention_mask)
        
        # Get predictions from image model
        img_logits = self.img_model(images=images)
        
        # Normalize attention weights
        normalized_weights = self.softmax(self.attention_weights)
        
        # Weighted sum of logits with attention
        weighted_text_logits = normalized_weights[0].unsqueeze(0) * text_logits
        weighted_img_logits = normalized_weights[1].unsqueeze(0) * img_logits
        
        logits = weighted_text_logits + weighted_img_logits
        
        return logits


class EnsembleLateFusion(nn.Module):
    """
    Ensemble of multiple late fusion models
    """
    def __init__(self, 
                 models,
                 weights=None):
        """
        Initialize the ensemble model
        
        Args:
            models (list): List of models to ensemble
            weights (list): List of weights for each model (optional)
        """
        super(EnsembleLateFusion, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        # If weights are not provided, use equal weighting
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights) / sum(weights)
    
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get predictions from all models
        all_logits = []
        for i, model in enumerate(self.models):
            logits = model(input_ids, attention_mask, images)
            all_logits.append(logits * self.weights[i])
        
        # Sum the weighted logits
        ensemble_logits = sum(all_logits)
        
        return ensemble_logits 