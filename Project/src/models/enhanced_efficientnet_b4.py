# CLARITy Enhanced EfficientNet-B4 Model
# src/models/enhanced_efficientnet_b4.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import math
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

class SpatialChannelSelfAttention(nn.Module):
    """Advanced Spatial-Channel-Self Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Self Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Spatial Attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        x_spatial = x * spatial_att
        
        # Channel Attention
        channel_att = self.channel_attention(x_spatial)
        x_channel = x_spatial * channel_att
        
        # Self Attention
        x_flat = x_channel.flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.self_attention(x_flat, x_flat, x_flat)
        x_self = attn_out.transpose(1, 2).reshape(b, c, h, w)
        
        # Feature Fusion
        x_fused = self.fusion(x_self)
        
        # Residual connection with learnable weight
        out = self.gamma * x_fused + x
        
        return out

class PyramidPooling(nn.Module):
    """Pyramid Pooling Module for multi-scale features"""
    
    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in pool_sizes
        ])
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = [x]
        
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        concatenated = torch.cat(pyramid_features, dim=1)
        out = self.final_conv(concatenated)
        
        return out

class EnhancedEfficientNetB4(nn.Module):
    """Enhanced EfficientNet-B4 with advanced attention and pyramid pooling"""
    
    def __init__(self, 
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            drop_rate=dropout_rate,
            drop_path_rate=0.2
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]  # Should be 1792 for B4
        
        # Advanced attention module
        self.attention = SpatialChannelSelfAttention(feature_dim)
        
        # Pyramid pooling
        self.pyramid_pooling = PyramidPooling(feature_dim)
        
        # Multi-scale feature extraction
        self.multiscale_conv = nn.ModuleList([
            nn.Conv2d(feature_dim, 256, 1),
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.Conv2d(feature_dim, 256, 5, padding=2),
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * 3 + feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Advanced classifier with residual connections
        classifier_dim = feature_dim * 2  # avg + max pooling
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ Enhanced EfficientNet-B4 Created:")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Advanced attention: ✅")
        print(f"   Pyramid pooling: ✅")
        print(f"   Multi-scale fusion: ✅")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """Initialize weights for optimal performance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Apply advanced attention
        attended_features = self.attention(features)
        
        # Apply pyramid pooling
        pyramid_features = self.pyramid_pooling(attended_features)
        
        # Multi-scale feature extraction
        multiscale_features = []
        for conv in self.multiscale_conv:
            multiscale_features.append(conv(pyramid_features))
        
        # Feature fusion
        all_features = torch.cat(multiscale_features + [pyramid_features], dim=1)
        fused_features = self.fusion_conv(all_features)
        
        # Global pooling
        avg_pool = self.global_avg_pool(fused_features)
        max_pool = self.global_max_pool(fused_features)
        
        # Concatenate and flatten
        pooled = torch.cat([avg_pool.flatten(1), max_pool.flatten(1)], dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# Advanced Loss Functions
class EnhancedFocalLoss(nn.Module):
    """Enhanced Focal Loss with minority class boosting"""
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.5,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1,
                 minority_boost: float = 5.0):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.minority_boost = minority_boost
        
        print(f"✅ Enhanced Focal Loss Created:")
        print(f"   Alpha: {alpha}")
        print(f"   Gamma: {gamma}")
        print(f"   Minority boost: {minority_boost}x")
        print(f"   Label smoothing: {label_smoothing}")
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Stable sigmoid
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
        
        # Enhanced focal loss
        focal_weight = (1 - probs) ** self.gamma
        pos_loss = -self.alpha * focal_weight * targets * torch.log(probs)
        
        neg_focal_weight = probs ** self.gamma
        neg_loss = -(1 - self.alpha) * neg_focal_weight * (1 - targets) * torch.log(1 - probs)
        
        loss = pos_loss + neg_loss
        
        # Apply enhanced class weights with minority boosting
        if self.class_weights is not None:
            enhanced_weights = self.class_weights.clone()
            
            # Boost minority classes (last few classes are typically rare)
            minority_indices = [-1, -2, -3]  # Hernia, Pleural_Thickening, Fibrosis
            for idx in minority_indices:
                enhanced_weights[idx] *= self.minority_boost
            
            loss = loss * enhanced_weights.unsqueeze(0).to(loss.device)
        
        return loss.mean()

# Factory Functions
def create_enhanced_efficientnet_b4(num_classes: int = 15, 
                                   class_weights: Optional[torch.Tensor] = None):
    """Create enhanced EfficientNet-B4 model"""
    
    model = EnhancedEfficientNetB4(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.4
    )
    
    return model

def create_enhanced_focal_loss(class_weights: Optional[torch.Tensor] = None,
                             minority_boost: float = 5.0):
    """Create enhanced focal loss with minority boosting"""
    
    return EnhancedFocalLoss(
        alpha=0.25,
        gamma=2.5,
        class_weights=class_weights,
        label_smoothing=0.1,
        minority_boost=minority_boost
    )
