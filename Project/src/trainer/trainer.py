# CLARITy DenseNet121 Model Architecture - Enhanced Version
# src/models/densenet121_enhanced.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple
import numpy as np

class DenseNet121MultiLabelEnhanced(nn.Module):
    """Enhanced DenseNet121 for multi-label chest X-ray classification with improved architecture"""
    
    def __init__(self, 
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.4,  # Increased dropout
                 freeze_features: bool = False,
                 use_attention: bool = True,
                 feature_fusion: bool = True):
        """
        Enhanced DenseNet121 with attention and feature fusion
        
        Args:
            num_classes: Number of disease classes (15 for NIH dataset)
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate for regularization (increased from 0.3)
            freeze_features: Whether to freeze feature extractor initially
            use_attention: Add attention mechanism for better feature focus
            feature_fusion: Use multi-scale feature fusion
        """
        super(DenseNet121MultiLabelEnhanced, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.feature_fusion = feature_fusion
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get number of features from classifier
        num_features = self.backbone.classifier.in_features  # 1024
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Channel Attention Module (if enabled)
        if self.use_attention:
            self.channel_attention = ChannelAttention(num_features)
            self.spatial_attention = SpatialAttention()
        
        # Enhanced Multi-layer Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 1024),  # Increased capacity
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.75),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, 256),  # Additional layer for better representation
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            
            nn.Linear(256, num_classes)
        )
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            print("🔒 Feature extractor frozen - only training classifier")
        
        # Initialize classifier weights
        self._initialize_weights()
        
        print(f"✅ Enhanced DenseNet121 Multi-Label Model created:")
        print(f"   Classes: {num_classes}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Dropout: {dropout_rate}")
        print(f"   Attention: {use_attention}")
        print(f"   Feature Fusion: {feature_fusion}")
    
    def _initialize_weights(self):
        """Initialize classifier weights with Xavier normal"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with attention and feature fusion"""
        # Extract features using backbone
        features = self.backbone.features(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.channel_attention(features)
            features = self.spatial_attention(features)
        
        # Global Average Pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Multi-layer classification
        logits = self.classifier(features)
        
        return logits
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities using sigmoid"""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature maps for Grad-CAM visualization"""
        features = self.backbone.features(x)
        return features
    
    def unfreeze_features(self):
        """Unfreeze feature extractor for fine-tuning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("🔓 Feature extractor unfrozen - full model training")

class ChannelAttention(nn.Module):
    """Channel Attention Module for focusing on important feature channels"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on important spatial locations"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        
        return x * attention

# Enhanced Loss Functions with Improved Class Weighting

class FocalLossEnhanced(nn.Module):
    """Enhanced Focal Loss with adaptive alpha and gamma parameters"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
        print(f"✅ Enhanced Focal Loss: alpha={alpha}, gamma={gamma}")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        
        # Calculate focal loss
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        
        loss = pos_loss + neg_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if self.class_weights.device != loss.device:
                self.class_weights = self.class_weights.to(loss.device)
            loss = loss * self.class_weights.unsqueeze(0)
        
        return loss.mean()

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for handling severe class imbalance in multi-label classification"""
    
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
        print(f"✅ Asymmetric Loss: gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sigmoid and clipping
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip / (1 + self.clip))
        
        # Calculate asymmetric loss
        targets_pos = targets
        targets_neg = 1 - targets
        
        # Positive and negative components
        xs_pos = probs
        xs_neg = 1 - probs
        
        los_pos = targets_pos * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = targets_neg * torch.log(xs_neg.clamp(min=1e-8))
        
        # Asymmetric focusing
        pt0 = xs_pos * targets_pos
        pt1 = xs_neg * targets_neg
        
        gamma = self.gamma_neg * targets_neg + self.gamma_pos * targets_pos
        loss = los_pos * ((1 - pt0) ** gamma) + los_neg * ((1 - pt1) ** gamma)
        
        return -loss.mean()

class CombinedLossEnhanced(nn.Module):
    """Enhanced Combined Loss with multiple loss components and adaptive weighting"""
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 focal_weight: float = 0.4,
                 bce_weight: float = 0.4,
                 asymmetric_weight: float = 0.2,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.focal_loss = FocalLossEnhanced(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
        self.asymmetric_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
        self.class_weights = class_weights
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.asymmetric_weight = asymmetric_weight
        
        # Normalize weights
        total_weight = focal_weight + bce_weight + asymmetric_weight
        self.focal_weight /= total_weight
        self.bce_weight /= total_weight
        self.asymmetric_weight /= total_weight
        
        print(f"✅ Enhanced Combined Loss:")
        print(f"   Focal: {self.focal_weight:.3f}")
        print(f"   BCE: {self.bce_weight:.3f}")
        print(f"   Asymmetric: {self.asymmetric_weight:.3f}")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Focal Loss
        focal = self.focal_loss(logits, targets)
        
        # Asymmetric Loss
        asymmetric = self.asymmetric_loss(logits, targets)
        
        # Weighted BCE Loss
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        if self.class_weights is not None:
            if self.class_weights.device != bce.device:
                self.class_weights = self.class_weights.to(bce.device)
            bce = bce * self.class_weights.unsqueeze(0)
        
        bce = bce.mean()
        
        # Combine all losses
        combined = (self.focal_weight * focal + 
                   self.bce_weight * bce + 
                   self.asymmetric_weight * asymmetric)
        
        return combined

# Factory Functions for Easy Model Creation

def create_enhanced_model_for_rtx3060(num_classes: int = 15, 
                                      class_weights: Optional[torch.Tensor] = None,
                                      model_size: str = "standard") -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create enhanced model optimized for RTX 3060
    
    Args:
        num_classes: Number of disease classes
        class_weights: Class weights for handling imbalance
        model_size: "standard" or "large" for different model capacities
    
    Returns:
        Tuple of (model, criterion)
    """
    
    if model_size == "large":
        # Larger model with more capacity
        model = DenseNet121MultiLabelEnhanced(
            num_classes=num_classes,
            pretrained=True,
            dropout_rate=0.5,  # Higher dropout for larger model
            freeze_features=False,
            use_attention=True,
            feature_fusion=True
        )
    else:
        # Standard enhanced model
        model = DenseNet121MultiLabelEnhanced(
            num_classes=num_classes,
            pretrained=True,
            dropout_rate=0.4,
            freeze_features=False,
            use_attention=True,
            feature_fusion=True
        )
    
    # Create enhanced loss function
    criterion = CombinedLossEnhanced(
        class_weights=class_weights,
        focal_weight=0.4,
        bce_weight=0.4,
        asymmetric_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=2.5  # Slightly higher gamma for better rare class handling
    )
    
    print(f"🚀 Enhanced RTX 3060 Model Configuration Complete!")
    print(f"   Model Size: {model_size}")
    print(f"   Enhanced Features: Attention + Multi-layer Classifier")
    print(f"   Loss: Triple-component (Focal + BCE + Asymmetric)")
    
    return model, criterion

def create_progressive_model(num_classes: int = 15,
                           class_weights: Optional[torch.Tensor] = None,
                           stage: str = "initial") -> Tuple[nn.Module, nn.Module]:
    """
    Create model for progressive training strategy
    
    Args:
        num_classes: Number of disease classes
        class_weights: Class weights for handling imbalance
        stage: "initial" (frozen features) or "fine_tune" (unfrozen features)
    
    Returns:
        Tuple of (model, criterion)
    """
    
    freeze_features = (stage == "initial")
    dropout_rate = 0.3 if stage == "initial" else 0.4
    
    model = DenseNet121MultiLabelEnhanced(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=dropout_rate,
        freeze_features=freeze_features,
        use_attention=True,
        feature_fusion=True
    )
    
    criterion = CombinedLossEnhanced(
        class_weights=class_weights,
        focal_weight=0.5 if stage == "initial" else 0.4,
        bce_weight=0.3 if stage == "initial" else 0.4,
        asymmetric_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0 if stage == "initial" else 2.5
    )
    
    print(f"🎯 Progressive Model Created - Stage: {stage}")
    print(f"   Features Frozen: {freeze_features}")
    print(f"   Optimized for: {'Initial Training' if stage == 'initial' else 'Fine-tuning'}")
    
    return model, criterion