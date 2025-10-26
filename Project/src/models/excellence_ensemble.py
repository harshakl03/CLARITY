# CLARITy High-Performance Ensemble Model Architecture
# src/models/excellence_ensemble.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SpatialChannelAttention(nn.Module):
    """Advanced Spatial-Channel Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, 
                                    padding=spatial_kernel//2, bias=False)
        
        # Self attention
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x).view(b, c))
        max_out = self.channel_mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(
            self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        )
        x = x * spatial_att
        
        # Self attention
        query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(b, -1, h * w)
        value = self.value(x).view(b, -1, h * w)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        out = self.gamma * out + x
        return out

class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion for better representation"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Different scale convolutions
        self.scale1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.scale2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.scale3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.scale4 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        fused = torch.cat([s1, s2, s3, s4], dim=1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.activation(fused)
        
        return fused

class ExcellenceEfficientNet(nn.Module):
    """Excellence EfficientNet with advanced features for 0.85+ AUC"""
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 use_advanced_attention: bool = True,
                 use_multiscale_fusion: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3, 4]  # Multi-scale features
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Multi-scale feature processing
        if use_multiscale_fusion:
            self.multiscale_fusion = nn.ModuleList([
                MultiScaleFeatureFusion(dim, 512) for dim in self.feature_dims
            ])
            fused_dim = 512 * len(self.feature_dims)
        else:
            fused_dim = self.feature_dims[-1]
        
        # Advanced attention
        if use_advanced_attention:
            self.attention = SpatialChannelAttention(fused_dim)
        
        # Global pooling with multiple strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Advanced classifier
        classifier_dim = fused_dim * 2  # avg + max pooling
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            
            nn.Linear(256, num_classes)
        )
        
        self.use_advanced_attention = use_advanced_attention
        self.use_multiscale_fusion = use_multiscale_fusion
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ Excellence EfficientNet Created:")
        print(f"   Backbone: {model_name}")
        print(f"   Advanced Attention: {use_advanced_attention}")
        print(f"   Multi-scale Fusion: {use_multiscale_fusion}")
        print(f"   Classifier dim: {classifier_dim}")
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        if self.use_multiscale_fusion:
            # Fuse multi-scale features
            fused_features = []
            for feature, fusion_layer in zip(features, self.multiscale_fusion):
                fused = fusion_layer(feature)
                # Resize to common size
                fused = F.interpolate(fused, size=features[-1].shape[2:], 
                                    mode='bilinear', align_corners=False)
                fused_features.append(fused)
            
            # Concatenate multi-scale features
            x = torch.cat(fused_features, dim=1)
        else:
            x = features[-1]
        
        # Apply advanced attention
        if self.use_advanced_attention:
            x = self.attention(x)
        
        # Global pooling with multiple strategies
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Flatten and concatenate
        x = torch.cat([
            avg_pool.flatten(1),
            max_pool.flatten(1)
        ], dim=1)
        
        # Classification
        logits = self.classifier(x)
        return logits

# Factory Functions

def create_excellence_efficientnet(num_classes: int = 15, 
                                 model_size: str = 'b4',
                                 class_weights: Optional[torch.Tensor] = None):
    """Create excellence EfficientNet model"""
    
    model_name = f'efficientnet_{model_size}'
    
    model = ExcellenceEfficientNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.5,
        use_advanced_attention=True,
        use_multiscale_fusion=True
    )
    
    return model

# Advanced Loss Functions for Excellence Performance

class ExcellenceLoss(nn.Module):
    """Advanced loss function for excellence performance"""
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.5,
                 label_smoothing: float = 0.1,
                 loss_weights: Dict[str, float] = None):
        super().__init__()
        
        self.class_weights = class_weights
        
        # Default loss weights for excellence
        if loss_weights is None:
            self.loss_weights = {
                'focal': 0.4,
                'asymmetric': 0.3,
                'bce': 0.2,
                'consistency': 0.1
            }
        else:
            self.loss_weights = loss_weights
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # FIXED PRINT STATEMENTS
        print(f"✅ Excellence Loss Function Created")
        print(f"   Focal weight: {self.loss_weights['focal']}")
        print(f"   Asymmetric weight: {self.loss_weights['asymmetric']}")
        print(f"   BCE weight: {self.loss_weights['bce']}")
        if 'consistency' in self.loss_weights:
            print(f"   Consistency weight: {self.loss_weights['consistency']}")
        else:
            print(f"   Components: {len(self.loss_weights)} (consistency disabled)")
    
    def focal_loss(self, logits, targets):
        """Enhanced focal loss"""
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        
        pos_loss = -self.focal_alpha * (1 - probs) ** self.focal_gamma * targets * torch.log(probs)
        neg_loss = -(1 - self.focal_alpha) * probs ** self.focal_gamma * (1 - targets) * torch.log(1 - probs)
        
        loss = pos_loss + neg_loss
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0).to(loss.device)
        
        return loss.mean()
    
    def asymmetric_loss(self, logits, targets):
        """Asymmetric loss for imbalanced data"""
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=0.05, max=0.95)
        
        xs_pos = probs
        xs_neg = 1 - probs
        
        los_pos = targets * torch.log(xs_pos)
        los_neg = (1 - targets) * torch.log(xs_neg)
        
        gamma_neg = 4.0
        gamma_pos = 1.0
        
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        
        loss = los_pos * ((1 - pt0) ** gamma_pos) + los_neg * ((1 - pt1) ** gamma_neg)
        
        return -loss.mean()
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Focal loss
        focal = self.focal_loss(logits, targets)
        
        # Asymmetric loss
        asymmetric = self.asymmetric_loss(logits, targets)
        
        # BCE loss
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        if self.class_weights is not None:
            bce = bce * self.class_weights.unsqueeze(0).to(bce.device)
        
        bce = bce.mean()
        
        # Combine losses (only use components that exist)
        total_loss = (
            self.loss_weights.get('focal', 0) * focal +
            self.loss_weights.get('asymmetric', 0) * asymmetric +
            self.loss_weights.get('bce', 0) * bce
        )
        
        return total_loss