# CLARITy Fast Stable Model Architecture
# src/models/fast_stable_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class StableFastEfficientNet(nn.Module):
    """Fast, stable EfficientNet optimized for speed and gradient stability"""
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b3',  # B3 instead of B4 for speed
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):  # Lower dropout
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load EfficientNet backbone (no features_only for simplicity)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Simple attention (lightweight)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Stable classifier with batch norm for gradient stability
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for stability
        self._initialize_weights()
        
        print(f"✅ Fast Stable EfficientNet Created:")
        print(f"   Backbone: {model_name}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Optimized for: Speed + Stability")
    
    def _initialize_weights(self):
        """Initialize weights for gradient stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for stability
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Simple attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# Simple Stable Loss Function
class StableFocalLoss(nn.Module):
    """Stable focal loss to prevent gradient explosion"""
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        print(f"✅ Stable Focal Loss Created:")
        print(f"   Alpha: {alpha}")
        print(f"   Gamma: {gamma}")
        print(f"   Label smoothing: {label_smoothing}")
    
    def forward(self, logits, targets):
        # Apply label smoothing for stability
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Stable sigmoid with clamping
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)  # Prevent log(0)
        
        # Focal loss computation
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        
        loss = pos_loss + neg_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0).to(loss.device)
        
        return loss.mean()

# Factory Functions
def create_fast_stable_model(num_classes: int = 15, 
                           model_size: str = 'b3',
                           class_weights: Optional[torch.Tensor] = None):
    """Create fast, stable model optimized for speed"""
    
    model_name = f'efficientnet_{model_size}'
    
    model = StableFastEfficientNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.3
    )
    
    return model

def create_stable_loss(class_weights: Optional[torch.Tensor] = None):
    """Create stable focal loss"""
    
    return StableFocalLoss(
        alpha=0.25,
        gamma=2.0,
        class_weights=class_weights,
        label_smoothing=0.1
    )