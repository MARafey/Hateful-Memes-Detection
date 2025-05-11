import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextOnlyModel(nn.Module):
    """
    Text-only model for hateful memes classification
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased', 
                 dropout=0.1,
                 freeze_text_model=False):
        """
        Initialize the text-only model
        
        Args:
            text_model_name (str): Name of the pretrained text model
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
        """
        super(TextOnlyModel, self).__init__()
        
        # Initialize transformer model
        self.config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name, config=self.config)
        
        # Freeze text model if specified
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        
        # Initialize weights
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
    
    def forward(self, input_ids, attention_mask, images=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Images (not used in this model)
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get text features from transformer model
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits

class TextWithPoolingModel(nn.Module):
    """
    Text model using pooled outputs for hateful memes classification
    """
    def __init__(self, 
                 text_model_name='bert-base-uncased', 
                 pooling_type='mean',
                 dropout=0.1,
                 freeze_text_model=False):
        """
        Initialize the text model with pooling
        
        Args:
            text_model_name (str): Name of the pretrained text model
            pooling_type (str): Type of pooling to use ('mean', 'max', or 'cls')
            dropout (float): Dropout probability
            freeze_text_model (bool): Whether to freeze the text model weights
        """
        super(TextWithPoolingModel, self).__init__()
        
        # Initialize transformer model
        self.config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name, config=self.config)
        
        # Freeze text model if specified
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Pooling type
        self.pooling_type = pooling_type
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        
        # Initialize weights
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
    
    def forward(self, input_ids, attention_mask, images=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Images (not used in this model)
            
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get text features from transformer model
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool the outputs based on the specified pooling type
        if self.pooling_type == 'mean':
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_type == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled_output = torch.max(token_embeddings, 1)[0]
        else:
            # CLS token pooling (default)
            pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits 