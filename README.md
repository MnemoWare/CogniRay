# CogniRay

Differentiable geometric memory for adaptive reasoning.

---

**CogniRay** is an experimental cognitive memory system built upon the principles of **Holographic Projection Memory (HPM)**.
It is designed as a structured, differentiable, and geometrically projective memory substrate that supports active inference,
directional recall, and adaptive semantic imprinting.

This repository contains:

* Theoretical foundations of Holographic Projection Memory (HPM) and CogniRay (see `docs/`)
* Minimal working implementation of HPM (read/write logic) (see `src/`)
* Examples and exploratory notebooks demonstrating usage (see `examples/`)
* *[Planned]* Core library for structured memory interfaces (see `cogniray/`)

---

## News  

**06/2025**:  
- **Experiment 01**: HPM converges rapidly from zero to target projections under exact rays — validating core update mechanics.  
- **Experiment 02**: HPM self-organizes from noise using only inferred projections — stable convergence, semantic drift, no backprop needed.  
- **Experiment 03**: HPM supports classical SGD — matches Delta-Learning performance under noisy rays, confirming architectural compatibility.  

---

## Key Concepts and Advantages

### **Geometric Access to Memory**

Memory is not accessed by keys or positions but by **directional projection rays** traversing a latent field. This enables:

* Content-based access via spatial geometry
* Localized attention along semantic trajectories
* View-dependent retrieval, mimicking perception

### **Differentiable Projection Mechanics**

Read and write operations are implemented as **smooth, differentiable integrals** along rays using soft kernels. This allows:

* Gradient-based learning of both memory and access patterns
* Seamless integration with neural architectures
* Full support for backpropagation and active error correction

### **Topological Divergence**

Conflicting updates do not overwrite but **diverge spatially**, forming distinct semantic regions. Benefits include:

* Natural separation of contradictory memories
* Controllable avoidance of catastrophic forgetting
* Self-organizing semantic clustering

### **Inference-Time Plasticity**

Memory is not static — it can **adapt on the fly** by modifying localized regions based on projection errors. This provides:

* Real-time correction of outdated or incomplete knowledge
* On-the-fly learning during inference
* Compatibility with lifelong and self-supervised learning

### **Multi-Scale Scanning**

Hierarchical, multi-resolution projection surfaces allow **focus from fine-grained detail to global context**, enabling:

* Coarse-to-fine reasoning
* Attention allocation to regions of semantic saliency
* Efficient memory search and reconstruction

### **Orientation-Aware Retrieval**

Projection direction vectors act as **semantic filters**, encoding preferences for angular alignment:

* Emulates biological orientation selectivity (e.g., V1 cortex)
* Enables disentangled recall based on directional cues
* Supports dynamic, learnable routing mechanisms

### **Interpretability and Visualizability**

Memory operations are explicitly geometric and spatially structured, which ensures:

* Transparent debugging and visualization of memory state
* Modular composability (e.g., stacking beams, splitting projection surfaces)
* Explainability for cognitive systems and symbolic overlays

### **Compatibility with Discrete and Continuous Models**

Though formulated in continuous space, HPM is implemented via voxelized tensors:

* Compatible with GPU-accelerated rasterization and grid processing
* Supports hybrid symbolic-geometric representations
* Bridges low-level neural fields and high-level memory primitives

---

## Roadmap

1. Stable research release (v0.1)
2. Core CogniRay module for PyTorch
3. Interactive visualization and ray tracing interface

---

## Project & Company

CogniRay is developed by **MnemoWare**, an independent research initiative building cognitive infrastructures for machine reasoning.

We design memory systems that learn by projecting, adapt by recalling, and scale with understanding.

> **MnemoWare. Shape your mind.**

Project: [cogniray.online](https://cogniray.online) *(coming soon)*  
Company: [mnemoware.com](https://mnemoware.com) *(coming soon)*  

Telegram group: [CogniRay](https://t.me/CogniRay)  
Telegram channel: [MnemoWare](https://t.me/MnemoWare)  

---

## Licensing

### **Documentation and Theoretical Materials**

Licensed under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)**
[https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to read, redistribute, and adapt the documentation for non-commercial purposes, provided that:

* Proper attribution is given (Konstantin Bashinsky)
* Derivative works are shared under the same license

### **Code and Implementations**

Licensed under the **CogniRay Non-Commercial License v1.0**
**Commercial use is strictly prohibited** without prior written permission from the author.

You may use the code for:

* Research
* Education
* Personal non-commercial experimentation

For commercial licensing, please contact: **[sombressoul@gmail.com](mailto:sombressoul@gmail.com)**

---

## Author

**Konstantin Bashinsky (a.k.a. Sombressoul)**  
AI researcher, memory reconstruction theorist, and occasional code-wrestler.
