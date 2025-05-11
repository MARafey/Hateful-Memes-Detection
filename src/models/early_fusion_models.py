import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import torchvision.models as models
import timm

class EarlyFusionModel(nn.Module):
    """
    Early fusion model for hateful memes classification that combines features before classification
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased',
                 img_model_name='resnet50',
                 fusion_method='concat',
                 dropout=0.1,
                 freeze_text_model=False,
                 freeze_img_model=False,
                 projection_dim=512):
        """
        Initialize the early fusion model
        
        Args:
            text_model_name (str): Name of the pretrained text model
            img_model_name (str): Name of the pretrained image model
            fusion_method (str): Method for fusion ('concat', 'sum', 'bilinear')
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
            freeze_img_model (bool): Whether to freeze the image model weights
            projection_dim (int): Dimension for feature projection (for 'sum' and 'bilinear' fusion)
        """
        super(EarlyFusionModel, self).__init__()
        
        # Initialize text model
        self.config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name, config=self.config)
        self.text_feature_dim = self.config.hidden_size
        
        # Freeze text model if specified
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Initialize image model based on name
        if img_model_name.startswith('resnet'):
            if img_model_name == 'resnet18':
                self.img_model = models.resnet18(pretrained=True)
                self.img_feature_dim = 512
            elif img_model_name == 'resnet34':
                self.img_model = models.resnet34(pretrained=True)
                self.img_feature_dim = 512
            elif img_model_name == 'resnet50':
                self.img_model = models.resnet50(pretrained=True)
                self.img_feature_dim = 2048
            elif img_model_name == 'resnet101':
                self.img_model = models.resnet101(pretrained=True)
                self.img_feature_dim = 2048
            elif img_model_name == 'resnet152':
                self.img_model = models.resnet152(pretrained=True)
                self.img_feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet model: {img_model_name}")
            
            # Remove the final classification layer
            self.img_model = nn.Sequential(*list(self.img_model.children())[:-1])
            
        elif img_model_name.startswith('efficientnet'):
            # Use timm for EfficientNet models
            self.img_model = timm.create_model(img_model_name, pretrained=True, num_classes=0)
            self.img_feature_dim = self.img_model.num_features
            
        elif img_model_name.startswith('vit'):
            # Use timm for Vision Transformer models
            self.img_model = timm.create_model(img_model_name, pretrained=True, num_classes=0)
            self.img_feature_dim = self.img_model.num_features
            
        else:
            raise ValueError(f"Unsupported image model: {img_model_name}")
        
        # Freeze image model if specified
        if freeze_img_model:
            for param in self.img_model.parameters():
                param.requires_grad = False
        
        # Fusion parameters
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        
        # Feature projection for text and image (if needed)
        if fusion_method in ['sum', 'bilinear']:
            self.text_projection = nn.Linear(self.text_feature_dim, projection_dim)
            self.img_projection = nn.Linear(self.img_feature_dim, projection_dim)
            
            if fusion_method == 'bilinear':
                self.bilinear = nn.Bilinear(projection_dim, projection_dim, projection_dim)
                self.classifier_dim = projection_dim
            else:  # sum
                self.classifier_dim = projection_dim
                
        else:  # concat fusion
            self.classifier_dim = self.text_feature_dim + self.img_feature_dim
        
        # Dropout and classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the weights of the linear layers
        """
        if hasattr(self, 'text_projection'):
            nn.init.normal_(self.text_projection.weight, std=0.02)
            nn.init.zeros_(self.text_projection.bias)
            
        if hasattr(self, 'img_projection'):
            nn.init.normal_(self.img_projection.weight, std=0.02)
            nn.init.zeros_(self.img_projection.bias)
            
        if hasattr(self, 'bilinear'):
            nn.init.normal_(self.bilinear.weight, std=0.02)
            nn.init.zeros_(self.bilinear.bias)
            
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
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
        # Get text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Use CLS token
        
        # Get image features
        img_features = self.img_model(images)
        img_features = img_features.reshape(img_features.size(0), -1)  # Flatten if needed
        
        # Feature fusion
        if self.fusion_method == 'concat':
            # Concatenate features
            fused_features = torch.cat([text_features, img_features], dim=1)
            
        elif self.fusion_method == 'sum':
            # Project to common space and sum
            text_projected = self.text_projection(text_features)
            img_projected = self.img_projection(img_features)
            fused_features = text_projected + img_projected
            
        elif self.fusion_method == 'bilinear':
            # Bilinear fusion
            text_projected = self.text_projection(text_features)
            img_projected = self.img_projection(img_features)
            fused_features = self.bilinear(text_projected, img_projected)
            
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Apply dropout and classification
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        
        return logits


class CrossAttentionFusion(nn.Module):
    """
    Early fusion model with cross-attention between text and image features
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased',
                 img_model_name='resnet50',
                 num_attention_heads=8,
                 dropout=0.1,
                 freeze_text_model=False,
                 freeze_img_model=False):
        """
        Initialize the cross-attention fusion model
        
        Args:
            text_model_name (str): Name of the pretrained text model
            img_model_name (str): Name of the pretrained image model
            num_attention_heads (int): Number of attention heads for cross-attention
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
            freeze_img_model (bool): Whether to freeze the image model weights
        """
        super(CrossAttentionFusion, self).__init__()
        
        # Initialize text model
        self.config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name, config=self.config)
        self.text_feature_dim = self.config.hidden_size
        
        # Freeze text model if specified
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Initialize image model
        if img_model_name.startswith('resnet'):
            if img_model_name == 'resnet18':
                base_model = models.resnet18(pretrained=True)
                self.img_feature_dim = 512
            elif img_model_name == 'resnet34':
                base_model = models.resnet34(pretrained=True)
                self.img_feature_dim = 512
            elif img_model_name == 'resnet50':
                base_model = models.resnet50(pretrained=True)
                self.img_feature_dim = 2048
            elif img_model_name == 'resnet101':
                base_model = models.resnet101(pretrained=True)
                self.img_feature_dim = 2048
            elif img_model_name == 'resnet152':
                base_model = models.resnet152(pretrained=True)
                self.img_feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet model: {img_model_name}")
            
            # Extract all layers except the final classification layer
            modules = list(base_model.children())[:-2]  # Keep the last conv layer
            self.img_model = nn.Sequential(*modules)
            
        else:
            raise ValueError(f"Currently only ResNet models are supported for CrossAttentionFusion")
        
        # Freeze image model if specified
        if freeze_img_model:
            for param in self.img_model.parameters():
                param.requires_grad = False
        
        # Project image features to match text feature dimension
        self.img_projection = nn.Conv2d(self.img_feature_dim, self.text_feature_dim, kernel_size=1)
        
        # Cross-attention parameters
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.text_feature_dim // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections for text-to-image attention
        self.text_query = nn.Linear(self.text_feature_dim, self.all_head_size)
        self.img_key = nn.Linear(self.text_feature_dim, self.all_head_size)
        self.img_value = nn.Linear(self.text_feature_dim, self.all_head_size)
        
        # Query, Key, Value projections for image-to-text attention
        self.img_query = nn.Linear(self.text_feature_dim, self.all_head_size)
        self.text_key = nn.Linear(self.text_feature_dim, self.all_head_size)
        self.text_value = nn.Linear(self.text_feature_dim, self.all_head_size)
        
        # Output projection
        self.attention_output = nn.Linear(self.all_head_size, self.text_feature_dim)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(self.text_feature_dim)
        self.layer_norm2 = nn.LayerNorm(self.text_feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.text_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize the weights
        """
        nn.init.normal_(self.img_projection.weight, std=0.02)
        if self.img_projection.bias is not None:
            nn.init.zeros_(self.img_projection.bias)
            
        for module in [self.text_query, self.img_key, self.img_value, 
                       self.img_query, self.text_key, self.text_value, 
                       self.attention_output]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
            
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def transpose_for_scores(self, x):
        """
        Reshape the tensor for multi-head attention
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Reshaped tensor for attention
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
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
        # Get text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get image features
        img_feature_maps = self.img_model(images)  # [batch_size, channels, height, width]
        
        # Project image features to match text feature dimension
        img_features_projected = self.img_projection(img_feature_maps)  # [batch_size, text_feature_dim, height, width]
        
        # Reshape image features to sequence format
        batch_size, feature_dim, height, width = img_features_projected.shape
        img_features = img_features_projected.permute(0, 2, 3, 1).reshape(batch_size, height * width, feature_dim)
        
        # Create attention mask for image features (all 1s)
        img_attention_mask = torch.ones(batch_size, height * width, device=input_ids.device)
        
        # Text-to-Image Cross-Attention
        # Project text features to query and image features to key/value
        mixed_query_layer = self.text_query(text_features)
        mixed_key_layer = self.img_key(img_features)
        mixed_value_layer = self.img_value(img_features)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate dot product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply attention mask
        # For text-to-image attention, we only need to mask the image features
        # attention_scores shape: [batch_size, num_heads, text_seq_len, img_seq_len]
        
        # Create proper mask: expand img_attention_mask to match attention_scores dimensions
        batch_size, num_heads, text_seq_len, img_seq_len = attention_scores.shape
        img_attention_mask_expanded = img_attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, text_seq_len, -1)
        
        # Apply text-to-image attention mask
        attention_scores = attention_scores.masked_fill(img_attention_mask_expanded == 0, -10000.0)
        
        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.attention_output(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Residual connection and layer normalization
        attention_output = self.layer_norm1(attention_output + text_features)
        
        # Take CLS token output for classification
        pooled_output = attention_output[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits 