"""
Model Definitions for Temporal Action Segmentation

This module contains the Multi-Stage Temporal Convolutional Network (MS-TCN)
model definitions for frame-level action segmentation on mouse behavior videos.

Based on:
- MS-TCN: "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation"
  (CVPR 2019) - https://arxiv.org/abs/1903.01945
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL: Multi-Stage Temporal Convolutional Network (MS-TCN)
# ============================================================================

class DilatedResidualLayer(nn.Module):
    """Dilated Residual Layer with temporal convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # Residual connection


class SingleStageModel(nn.Module):
    """Single stage of the MS-TCN model"""
    
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_f_maps, num_f_maps, kernel_size=3, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MS_TCN(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network
    
    Args:
        num_stages: Number of stages (default: 4)
        num_layers: Number of layers per stage (default: 10)
        num_f_maps: Number of feature maps (default: 64)
        dim: Input feature dimension (default: 2048 for ResNet50)
        num_classes: Number of action classes
    """
    
    def __init__(self, num_stages=4, num_layers=10, num_f_maps=64, dim=2048, num_classes=2):
        super().__init__()
        self.num_stages = num_stages
        
        # First stage takes original features
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        
        # Subsequent stages take predictions from previous stage
        self.stages = nn.ModuleList([
            SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
            for _ in range(num_stages - 1)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, feature_dim, sequence_length)
        
        Returns:
            List of predictions from each stage (batch_size, num_classes, sequence_length)
        """
        outputs = []
        
        # Stage 1
        out = self.stage1(x)
        outputs.append(out)
        
        # Subsequent stages
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        
        return outputs

