# SPDX-License-Identifier: Apache-2.0
import torch
from src.alms import ALMSLayer

def test_import():
    """Test that ALMSLayer can be imported"""
    layer = ALMSLayer()
    assert layer.k == 32
    assert layer.lambda_reg == 0.1

def test_forward():
    """Test basic forward pass"""
    layer = ALMSLayer()
    features = torch.randn(32, 1024)
    enhanced, graph = layer(features, return_graph=True)
    
    assert enhanced.shape == (32, 1024)
    assert graph.shape == (32, 32)
