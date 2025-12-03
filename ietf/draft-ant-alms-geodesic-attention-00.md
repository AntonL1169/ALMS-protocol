
#### Содержимое:
```markdown
---
title: ALMS — Geodesic Attention for Similarity Fields
abbrev: ALMS
docname: draft-ant-alms-geodesic-attention-00
category: info
ipr: trust200902
area: ART
kw: Privacy, Attention, Topology
author:
 -
    ins: L. Antonov
    name: Anton Lyubimov
    org: Independent
    email: antonl110569@gmail.com
abstract: |
  This document describes ALMS, a geodesic attention mechanism that enhances
  similarity-based representations via topological diffusion on Riemannian
  manifolds. It is compatible with differential privacy and operates in
  high-dimensional sparse spaces.
---

# Introduction

Traditional attention mechanisms rely on Euclidean similarity metrics, which fail to capture manifold structure in high-dimensional sparse spaces. ALMS introduces a geodesic attention layer that performs topological diffusion on a k-nearest-neighbor graph built on cosine similarity. The method is differentiable, privacy-aware, and robust to batch size.

# Architecture

## Graph Construction
Builds a symmetric kNN graph using cosine similarity.

## Harmonic Potential Diffusion
Approximates geodesic distances via 2-step random walk with normalized Laplacian.

## Attention Weights
Uses inverse geodesic distance with temperature scaling.

# Privacy Considerations

Adds Gaussian noise (ε=0.01) to landmark dimensions during training.

# Security Considerations

No additional attack surfaces.

# IANA Considerations

This document has no IANA actions.

--- appendix

# Reference Implementation

Open-source: https://github.com/AntonL1169/alms-protocol
