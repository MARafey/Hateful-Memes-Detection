import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ImageOnlyModel(nn.Module):
    """
    Image-only model for hateful memes classification
    """
    def __init__(self, 
                 img_model_name='resnet50', 
                 pretrained=True,
                 dropout=0.1,
                 freeze_img_model=False):
        """
        Initialize the image-only model
        
        Args:
            img_model_name (str): Name of the pretrained image model
            pretrained (bool): Whether to use pretrained weights
            dropout (float): Dropout probability
            freeze_img_model (bool): Whether to freeze the image model weights
        """
        super(ImageOnlyModel, self).__init__()
        
        # Initialize image model based on name
        if img_model_name.startswith('resnet'):
            if img_model_name == 'resnet18':
                self.img_model = models.resnet18(pretrained=pretrained)
                self.feature_dim = 512
            elif img_model_name == 'resnet34':
                self.img_model = models.resnet34(pretrained=pretrained)
                self.feature_dim = 512
            elif img_model_name == 'resnet50':
                self.img_model = models.resnet50(pretrained=pretrained)
                self.feature_dim = 2048
            elif img_model_name == 'resnet101':
                self.img_model = models.resnet101(pretrained=pretrained)
                self.feature_dim = 2048
            elif img_model_name == 'resnet152':
                self.img_model = models.resnet152(pretrained=pretrained)
                self.feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet model: {img_model_name}")
            
            # Remove the final classification layer
            self.img_model = nn.Sequential(*list(self.img_model.children())[:-1])
            
        elif img_model_name.startswith('efficientnet'):
            # Use timm for EfficientNet models
            self.img_model = timm.create_model(img_model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.img_model.num_features
            
        elif img_model_name.startswith('vit'):
            # Use timm for Vision Transformer models
            self.img_model = timm.create_model(img_model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.img_model.num_features
            
        else:
            raise ValueError(f"Unsupported image model: {img_model_name}")
        
        # Freeze image model if specified
        if freeze_img_model:
            for param in self.img_model.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, 2)
        
        # Initialize weights of the classifier
        self._init_weights(self.classifier)
    
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
    
    def forward(self, input_ids=None, attention_mask=None, images=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (not used in this model)
            attention_mask: Attention mask (not used in this model)
            images: Images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get image features
        img_features = self.img_model(images)
        
        # Flatten features if needed
        if len(img_features.shape) > 2:
            img_features = img_features.reshape(img_features.size(0), -1)
        
        # Apply dropout and classification head
        img_features = self.dropout(img_features)
        logits = self.classifier(img_features)
        
        return logits


class ResNetWithAttention(nn.Module):
    """
    ResNet model with attention mechanism for hateful memes classification
    """
    def __init__(self, 
                 img_model_name='resnet50', 
                 pretrained=True,
                 dropout=0.1,
                 freeze_img_model=False):
        """
        Initialize the ResNet model with attention
        
        Args:
            img_model_name (str): Name of the pretrained ResNet model
            pretrained (bool): Whether to use pretrained weights
            dropout (float): Dropout probability
            freeze_img_model (bool): Whether to freeze the image model weights
        """
        super(ResNetWithAttention, self).__init__()
        
        # Initialize ResNet model based on name
        if img_model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif img_model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif img_model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif img_model_name == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif img_model_name == 'resnet152':
            base_model = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {img_model_name}")
        
        # Extract all layers except the final classification layer
        modules = list(base_model.children())[:-2]  # Keep the last conv layer
        self.img_model = nn.Sequential(*modules)
        
        # Freeze image model if specified
        if freeze_img_model:
            for param in self.img_model.parameters():
                param.requires_grad = False
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, 2)
        
        # Initialize weights
        self._init_weights(self.attention)
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """
        Initialize the weights
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids=None, attention_mask=None, images=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (not used in this model)
            attention_mask: Attention mask (not used in this model)
            images: Images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get feature maps from the image model
        feature_maps = self.img_model(images)  # Shape: [batch_size, channels, height, width]
        
        # Apply attention
        attention_weights = self.attention(feature_maps)  # Shape: [batch_size, 1, height, width]
        attended_feature_maps = feature_maps * attention_weights  # Shape: [batch_size, channels, height, width]
        
        # Global average pooling
        pooled_features = self.avg_pool(attended_feature_maps)  # Shape: [batch_size, channels, 1, 1]
        pooled_features = pooled_features.reshape(pooled_features.size(0), -1)  # Shape: [batch_size, channels]
        
        # Apply dropout and classification head
        pooled_features = self.dropout(pooled_features)
        logits = self.classifier(pooled_features)
        
        return logits 