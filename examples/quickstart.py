#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Quickstart: Zero-shot shape from audio

import torch
from src.alms import ALMSLayer

# 1. Load pre-trained ALMS layer
layer = ALMSLayer(k=32, lambda_reg=0.1)

# 2. Create unified features (audio + shape)
# Audio: first 512 dims, Shape: last 512 dims
audio_feat = torch.randn(128, 512)  # From your audio encoder
shape_feat = torch.randn(128, 512)  # From PointNet
features = torch.cat([audio_feat, shape_feat], dim=1)

# 3. Run ALMS
enhanced, graph = layer(features, return_graph=True)

# 4. Zero-shot retrieval
similarity = enhanced[:, :512] @ enhanced[:, 512:].T
predicted = similarity.argmax(dim=1)

print(f"Predicted shapes: {predicted[:5]}")
print(f"Graph density: {graph.nonzero().shape[0] / graph.shape[0]:.2%}")
