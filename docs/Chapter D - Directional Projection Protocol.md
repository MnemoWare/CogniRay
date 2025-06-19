# Chapter D - HPM - Directional Projection Protocol

> *Mathematical formulation and implementation guidelines for coherent directional projection in Holographic Projection Memory (HPM) systems.*

---

## D.0 Dimensionality Perspective and Scalability Strategy

The Holographic Projection Memory (HPM) mechanism operates over a differentiable memory field $W(x)$, where $x \in \mathbb{R}^N$. While the model is mathematically valid for arbitrary $N$, a key architectural insight lies in recognizing that increasing the dimensionality $N$ is not required for scaling memory capacity or expressivity.

### Fixed-Dimensional Memory with Scalable Projections

Instead of increasing $N$, the HPM system can scale by manipulating the projection interface:

* **Projection resolution $R$:** the number of sampling points $u$ on the hypersurface $\mathcal{P}$
* **Projection orientations $\{ \mathbf{v}_m \}_{m=1}^{M}$:** number of distinct directional beams

The total informational capacity of the system (in terms of independent projection modes) can be approximated as:

$$
\text{Capacity}_{\text{proj}} \sim O(R^{N-1} \cdot M)
$$

Where:

* $R^{N-1}$ is the resolution of the projection surface in $(N-1)$ dimensions
* $M$ is the number of angular directions (e.g., $M = 16\dots 64$)

This implies that **the effective degrees of freedom grow multiplicatively** with projection density and angular coverage - even when $N$ is fixed (e.g., $N = 3$).

### Practical Implications

* **3D memory ($N=3$) is sufficient** for most tasks when paired with dense 2D projection surfaces
* Modern GPUs support volumetric tensors and ray sampling efficiently in $\mathbb{R}^3$
* Projection operations are parallelizable over $R$ and $M$

### Benefits over High-Dimensional Architectures

| Classical Scaling | HPM Projection Scaling       |
| ----------------- | ---------------------------- |
| Increase $N$    | Increase $R$, $M$        |
| $O(D^N)$ memory | $O(D^3 \cdot R^2 \cdot M)$ |
| Sparse gradients  | Dense, localized updates     |

### High-Dimensional Extensions Remain Valid

Although growth in $N$ is unnecessary for scaling, it remains **fully supported**. For multimodal memory (e.g., spatial-temporal or semantic-visual embeddings), higher $N$ allows native disentanglement. However, even in these cases, directional projection remains the core retrieval and modulation strategy.

> **Conclusion:** HPM decouples memory capacity from spatial dimensionality. By scaling projection resolution and directionality, the system enables efficient access to distributed memory with fixed physical dimensionality - allowing for stable, interpretable, and tractable memory operations.

---

## D.1 Directional Ray Model

In the Directional Projection Protocol, each projection is defined by a ray emitted from a point on a projection hypersurface $\mathcal{P}(u) \subset \mathbb{R}^N$, parameterized by $u \in \mathbb{R}^{N-1}$. The projection ray is constructed as:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}, \quad t \in \mathbb{R}
$$

where:

* $\Phi(u) \in \mathbb{R}^N$ maps the projection coordinate $u$ to a location in memory space,
* $\mathbf{v} \in \mathbb{R}^N$ is the global projection direction shared across all $u$,
* $\ell_u(t)$ defines the coherent beam line associated with projection point $u$.

This formulation defines a **coherent, parallel beam** architecture, where all rays are aligned and maintain fixed directional geometry.

> **Note:** While most practical implementations use a shared direction vector $\mathbf{v}$ for the entire projection bundle, the model fully supports the general case where each ray may have its own direction $\mathbf{v}_u$. All projection formulas and kernels remain valid under this extension, provided $\mathbf{v}_u$ is differentiable with respect to $u$ or externally defined.

---

### Norm Regularization of $\mathbf{v}$

To ensure well-posedness of the projection model and numerical stability during learning, the projection direction $\mathbf{v}$ must remain non-degenerate:

$$
\| \mathbf{v} \|_2 \geq \varepsilon, \quad \text{with } \varepsilon > 0 \text{ (e.g., } 10^{-4} \text{)}
$$

During optimization, $\mathbf{v}$ is explicitly normalized:

$$
\mathbf{v} \leftarrow \frac{\mathbf{v}}{\max(\|\mathbf{v}\|_2, \varepsilon)}
$$

Additionally, a soft constraint term may be added to the loss:

$$
\mathcal{L}_{\text{norm}} = \lambda_{\text{norm}} \cdot (1 - \|\mathbf{v}\|_2)^2
$$

This maintains directional stability and prevents projection collapse during gradient-based learning.

---

### Angular Selectivity and Orientation Tuning

To encode directional sensitivity in projection and retrieval, we introduce the concept of **angular selectivity** via orientation tuning. Let:

* $\theta = \angle(\mathbf{v}, \mathbf{v}_{\text{pref}})$ be the angle between the current projection direction and a preferred direction,
* $\kappa > 0$ control the sharpness of selectivity,
* $S(\theta)$ define the angular response profile.

Then:

$$
S(\theta) = \exp(-\kappa \cdot \theta^2)
$$

This can be used to:

* Weigh the contribution of a projection based on its alignment with target memory structures,
* Modulate update magnitudes $\delta(u)$ or sampling weights based on orientation,
* Emulate biologically inspired tuning curves (e.g., in primary visual cortex).

Angular selectivity also underpins advanced routing and disentanglement mechanisms, particularly when multiple preferred orientations $\{ \mathbf{v}_m \}_{m=1}^M$ are defined across the system.

---

> The directional ray model defines the geometric and functional basis of HPM projection. By enforcing norm stability and introducing orientation-aware modulation, it establishes a flexible yet robust mechanism for encoding and interacting with distributed memory structures.

---

## D.2 Attenuation Model Along the Ray

In directional projection, each ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$ extends from a point on the projection surface $\mathcal{P}(u) \subset \mathbb{R}^N$ into the memory field $W(x)$. As information is gathered along this ray, its contribution must account not only for spatial proximity, but also for **attenuation over distance** - modeling the diminishing influence of deeper memory regions.

We define the **longitudinal attenuation** function as:

$$
A(t) = \exp\left( -\frac{t}{\tau} \right), \quad t \geq 0
$$

Where:

* $t$ is the signed distance along the ray from the origin $\Phi(u)$,
* $\tau > 0$ is the **attenuation constant**, controlling how quickly the beam decays with depth.

This attenuation is applied multiplicatively within the full kernel $K(x, \ell_u)$, combining with lateral distance sensitivity (discussed in Section D.3).

---

### Interpretation of $\tau$ as Contextual Focus

The parameter $\tau$ governs how far the projection ray "sees" into the memory volume:

* **Small $\tau$**:

  * Rapid decay of ray intensity
  * Emphasis on **local features** near the surface
  * Useful for **fine-grained detail detection**

* **Large $\tau$**:

  * Slow decay, deep ray penetration
  * Aggregation over **broad semantic regions**
  * Enables **contextual integration** and **global reasoning**

This mirrors **attention mechanisms** in transformer models:

* $\tau$ acts as a soft span parameter
* The projection acts as a directionally constrained receptive field

---

### Dynamic Attenuation: $\tau(u)$

To enable **adaptive focus**, the attenuation parameter can vary across the projection surface:

$$
\tau = \tau(u)
$$

This allows the model to dynamically modulate depth sensitivity based on location or learned semantic cues. For example:

* Shallow focus in cluttered regions
* Deep integration in smooth or low-density zones

In practice, $\tau(u)$ can be:

* Learned as an auxiliary output of $\Phi(u)$
* Conditioned on external input or attention control
* Regularized to maintain continuity and avoid oscillations

---

### Transition to Contribution Kernel

The attenuation function $A(t) = \exp(-t / \tau)$ defines the **longitudinal weighting** of contributions along the ray. Combined with lateral Gaussian weighting (based on perpendicular distance $d_\perp$), it forms the full projection kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

Where:

* $t_x$ is the value of $t$ minimizing $\| x - \ell_u(t) \|^2$
* $\sigma$ is the beam's lateral width (see Section D.3)

This kernel determines both the forward projection (Section D.3) and gradient updates (Section D.4).

> By interpreting $\tau$ as a geometric control over contextual range, the attenuation model endows HPM with selective depth-awareness - enabling the system to adjust its semantic field of view while preserving full differentiability.

---

## D.3 Kernel-Based Contribution of Memory Points

Each projection in HPM aggregates contributions from the memory field $W(x)$ along a directional ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$. To determine the influence of any memory point $x \in \mathbb{R}^N$ on a given projection coordinate $u$, we define a continuous, differentiable kernel $K(x, \ell_u)$ that weights contributions based on geometric proximity.

---

### Projection-Aligned Coordinates

Let $x \in \mathbb{R}^N$ be a candidate point in memory. We compute:

1. **Axial projection onto the ray**:

$$
t_x = (x - \Phi(u)) \cdot \mathbf{v}
$$

This gives the location on the ray $\ell_u(t)$ closest to $x$. It is the scalar component of $x - \Phi(u)$ in the direction of $\mathbf{v}$.

2. **Reconstructed point on the ray**:

$$
\hat{x} = \Phi(u) + t_x \cdot \mathbf{v}
$$

3. **Perpendicular distance to the ray**:

$$
d_\perp = \| x - \hat{x} \|
$$

---

### Isotropic Contribution Kernel

The standard contribution kernel used in projection and update is defined as:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

Where:

* $\sigma > 0$ is the lateral beam width,
* $\tau > 0$ is the attenuation constant (from Section D.2).

This formulation ensures:

* **Locality**: Only points near the beam axis significantly contribute.
* **Causal attenuation**: Points deeper along the ray have lower influence.

---

### Anisotropic Generalization

In high-capacity or highly structured settings, it may be beneficial to use **anisotropic kernels** that allow separate control over lateral and longitudinal sensitivity. Define:

$$
K(x, \ell_u) = \exp\left( -\frac{d_\perp^2}{2\sigma_\perp^2} - \frac{t_x^2}{2\sigma_\parallel^2} \right)
$$

Where:

* $\sigma_\perp$ governs lateral focus (orthogonal to $\mathbf{v}$),
* $\sigma_\parallel$ controls axial integration along the ray.

This kernel maintains full differentiability and allows for finer tuning of projection behavior:

* Narrow $\sigma_\perp$: highly localized spatial precision.
* Wide $\sigma_\parallel$: deeper aggregation.

---

### Final Projection Expression

The projection value at coordinate $u$ is computed by integrating the contributions of all memory points:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

In practice, this is approximated over a discretized volume, often using sampled rays, local neighborhoods, or beam-aligned convolution windows.

---

### Transition to Gradient Computation

The structure of $K(x, \ell_u)$ ensures that the projection operator $T(u)$ is differentiable with respect to both the memory field $W(x)$ and projection parameters. This enables gradient flow through directional beams, which is addressed in the next section.

> The contribution kernel defines how memory structure interacts with directional perception. By combining geometrically grounded alignment with soft attenuation and adaptable resolution, HPM achieves controlled, expressive memory access in differentiable form.

---

## D.4 Gradient Computation

The projection operator in HPM is fully differentiable with respect to both the memory field $W(x)$ and the projection geometry parameters $\Phi(u)$ and $\mathbf{v}$. This section details the derivation of gradients needed for end-to-end learning.

---

### Gradient with Respect to Memory Field $W(x)$

The projected value at point $u$ is:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

As $K(x, \ell_u)$ is fixed for given $u$, we have:

$$
\frac{\partial T(u)}{\partial W(x)} = K(x, \ell_u)
$$

This establishes a direct correspondence between projection and contribution kernel - facilitating sparse, localized gradient propagation.

---

### Gradient with Respect to Projection Origin $\Phi(u)$

Since $t_x = (x - \Phi(u)) \cdot \mathbf{v}$ and $d_\perp = \| x - (\Phi(u) + t_x \cdot \mathbf{v}) \|$, both are functions of $\Phi(u)$, and thus:

$$
\nabla_{\Phi} T(u) = \int W(x) \cdot \nabla_{\Phi} K(x, \ell_u) \, dx
$$

The derivative $\nabla_{\Phi} K$ involves chain rule through $t_x$ and $d_\perp$. Exact forms depend on kernel shape (see D.3), but remain analytically tractable.

---

### Gradient with Respect to Direction Vector $\mathbf{v}$

We now derive $\nabla_{\mathbf{v}} T(u)$. Using:

$$
t_x = (x - \Phi(u)) \cdot \mathbf{v}, \quad \hat{x} = \Phi(u) + t_x \cdot \mathbf{v}, \quad d_\perp = x - \hat{x}
$$

Then, the kernel:

$$
K(x, \ell_u) = \exp\left( -\frac{\|d_\perp\|^2}{2\sigma^2} \right) \cdot \exp\left( -\frac{t_x}{\tau} \right)
$$

Its gradient w\.r.t. $\mathbf{v}$ is:

$$
\nabla_{\mathbf{v}} K = K \cdot \left( \frac{1}{\sigma^2} \cdot d_\perp^T \cdot \nabla_{\mathbf{v}} d_\perp - \frac{1}{\tau} \cdot \nabla_{\mathbf{v}} t_x \right)
$$

With:

$$
\nabla_{\mathbf{v}} t_x = x - \Phi(u)
$$

$$
\nabla_{\mathbf{v}} \hat{x} = t_x \cdot I + \mathbf{v} \otimes (x - \Phi(u))
$$

$$
\nabla_{\mathbf{v}} d_\perp = -\nabla_{\mathbf{v}} \hat{x}
$$

These components yield a fully differentiable path for updating $\mathbf{v}$ through gradient descent.

---

### Implementation Example (Pseudocode)

```python
# Given x, Phi, v
r = x - Phi

# Projection scalar along v
t_x = r.dot(v)

# Closest point on ray
x_hat = Phi + t_x * v

# Perpendicular displacement
d_perp = x - x_hat

# Contribution kernel
K = exp(-d_perp.norm()**2 / (2 * sigma**2)) * exp(-t_x / tau)
```

This expression can be batched over many rays and memory points, supporting efficient GPU implementation.

---

### Transition to Local Update Rule

As the kernel gradient structure is simple and localized, it permits not only backpropagation-based optimization, but also **direct local updates** without requiring full network-level gradient flows. This will be the focus of Section D.5.

> The gradient framework in HPM provides a precise and tractable pathway for optimizing both content and structure. Through analytic expressions for $t_x$, $d_\perp$, and their derivatives, directional projection becomes a powerful tool for memory interaction and learning.

---

## D.5 Local Update Without Backpropagation

In certain scenarios, particularly during inference or interaction, it is desirable to modify the memory field $W(x)$ in real time, based on projection-level error signals, without requiring global backpropagation. HPM supports such **local, direct updates** by leveraging the differentiable structure of the directional projection kernel.

---

### Update Rule

Suppose we observe a projection discrepancy:

$$
\delta(u) = T^*(u) - T(u)
$$

where $T^*(u)$ is the desired or target projection value at coordinate $u$. Then, the memory field can be updated as:

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u)
$$

Where:

* $\alpha > 0$ is a learning rate or imprinting coefficient,
* $K(x, \ell_u)$ is the projection kernel defined in Section D.3,
* The update is applied locally and additively.

This rule requires only the projection error $\delta(u)$ and kernel values - no global loss or gradient computation is needed.

---

### Relation to Chapter A: Active Inference

This mechanism directly instantiates the concept of **inference-time plasticity** described in Chapter A.3. There, a projection error modifies the memory field through its localized footprint. In the directional case:

* The projection kernel $K(x, \ell_u)$ is no longer isotropic.
* The beam direction $\mathbf{v}$ introduces **contextual selectivity**.

Thus, the update becomes **spatially and directionally aware**, enabling **modulated self-imprinting** with semantic targeting.

---

### Relation to Chapter B: Associative Update Dynamics

Chapter B formalizes associative memory updates via smooth back-projection:

$$
\Delta W(x) = \int \delta(u) \cdot K(x, \ell_u) \, du
$$

The local rule here is a special case:

* It updates $W(x)$ for a single $u$ at a time.
* Direction $\mathbf{v}$ constrains the region of influence.

This introduces an implicit **orientation-based generalization filter**: updates affect only memory locations that align with the beam geometry. If multiple orientations are encoded (as in Section D.1), update propagation remains restricted to **semantically aligned subspaces**.

---

### Efficiency and Safety

Advantages of this method include:

* **Real-time applicability** in deployed systems
* **No gradient tracking overhead**
* **Soft spatial locality**
* **Compatibility with continual learning and adaptation**

To prevent drift or instability:

* $\alpha$ should decay over time or be gated by confidence
* Updates can be normalized by $\int K(x, \ell_u) dx$
* Multiple $u$ values can be blended with weighted averaging

---

### Transition to Cognitive Interpretation

The directional local update mechanism can be interpreted as an **introspective correction**: the system modulates its own internal structure in response to external mismatch, guided by spatial alignment and directional expectation.

This opens the door to biologically plausible memory dynamics, as discussed in Section D.6.

> The local update rule operationalizes fast, semantically aligned memory adaptation - bridging inference, learning, and memory within a unified geometric framework.

---

## D.6 Cognitive and Physical Interpretation

Beyond its mathematical formulation, the directional projection mechanism in HPM carries deep analogies with both biological cognition and physical sensing. This section highlights two key perspectives - one grounded in **neuroscience**, the other in **optical physics** - that provide intuitive grounding and motivation for the system's design.

---

### 1. Neurocognitive Analogy: Orientation-Selective Memory Access

The architecture of HPM, particularly with a directional kernel and orientation-tuned update rules, bears strong resemblance to functional organization in early visual cortex (V1):

* In V1, neurons are organized into **orientation columns**, each responding maximally to specific edge directions.
* The directional vector $\mathbf{v}$ in HPM plays the role of a **preferred orientation**.
* The angular response function:

$$
S(\theta) = \exp(-\kappa \cdot \theta^2)
$$

acts as a **tuning curve**, analogous to neural selectivity profiles observed in biological systems.

* Directionally structured updates and projections emulate **receptive fields** shaped by task and stimulus context.
* Learning dynamics (e.g., Section D.5) resemble **experience-dependent plasticity** - selectively reinforcing responses along frequently used directions.

> Directional projection thus encodes not just spatial but **orientational context**, enabling biologically plausible semantic routing through geometry.

---

### 2. Physical Analogy: Attenuated Light Propagation

The attenuation function:

$$
A(t) = \exp\left(-\frac{t}{\tau}\right)
$$

models light propagation through an absorbing medium - a principle well-understood in optics and radiative transfer. In this view:

* $\tau$ corresponds to the **absorption length** - how far a beam can penetrate before decaying.
* The beam $\ell_u(t)$ becomes a **probe** that samples the medium with gradually vanishing influence.
* The memory field $W(x)$ is analogous to a **semi-transparent volume** - a "fog of meaning" through which information must be recovered.

This analogy justifies the use of exponential decay as both **physically motivated** and **computationally tractable**, linking directional depth sensitivity to perception.

> The system performs metaphorical tomography - sweeping through representational space to reconstruct latent structure.

---

### Combined View

In both metaphors:

* Directionality $\mathbf{v}$ modulates semantic access
* Depth $t$ and decay $\tau$ regulate context span
* Angular selectivity $S(\theta)$ implements filtering and prioritization

Together, these components make directional HPM more than a projection mechanism - they render it an **active perception operator** over distributed memory, capable of structured introspection and adaptively routed generalization.

> HPM directional projection mirrors vision not only in geometry, but in function: it sees, filters, adapts, and remembers - with structure.

---

## D.7 Theoretical Extensions

This section explores formal consequences and extensions of the directional projection model introduced in HPM, focusing on how angular orientation affects cluster dynamics, memory separability, and semantic plasticity. These results complement the conflict-resolution mechanisms described in Chapter C.

---

### 1. Angularly Modulated Cluster Divergence

Let $\rho_1(x), \rho_2(x)$ be two memory clusters modeled as Gaussian fields:

$$
\rho_i(x) = \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)
$$

Let $\mathbf{v}_1, \mathbf{v}_2$ be the projection directions associated with updates to $\rho_1$ and $\rho_2$, respectively. Denote:

* $\theta = \angle(\mathbf{v}_1, \mathbf{v}_2) \in [0, \pi]$ - angular mismatch
* $F_{ij}(\theta)$ - effective repulsive force between clusters

We propose that angular misalignment increases semantic divergence. Specifically:

> **Theorem 2 (Angular Divergence Acceleration).**
>
> Let $x_1(t), x_2(t) \in \mathbb{R}^N$ evolve under repulsion derived from directional updates along $\mathbf{v}_1, \mathbf{v}_2$. Then for fixed $\sigma > 0$, there exists a critical angle $\theta_{\text{crit}} \in (0, \pi)$ such that:
>
> $$
> \theta > \theta_{\text{crit}} \quad \Rightarrow \quad \frac{d}{dt} \|x_1 - x_2\| \text{ increases monotonically}
> $$

**Sketch of Justification:**

* Directional updates generate kernels with aligned beam paths.
* Overlapping updates $K_1(x), K_2(x)$ produce interference terms decaying with $\cos(\theta)$.
* For $\theta > \theta_{\text{crit}}$, the overlap between kernels becomes negligible, maximizing net gradient difference and hence separation rate.

This effect enhances **topological divergence** (Chapter C.10) by introducing angular disentanglement.

---

### 2. Angular Sensitivity Derivative

Recall the directional kernel:

$$
K(x, \ell_u) = \exp\left(-\frac{d_\perp^2}{2\sigma^2}\right) \cdot \exp\left(-\frac{t_x}{\tau}\right)
$$

Let $\theta$ be the angle between the current direction $\mathbf{v}$ and a preferred orientation $\mathbf{v}_{\text{pref}}$. Define the **angular tuning function**:

$$
S(\theta) = \exp(-\kappa \cdot \theta^2)
$$

Then, the second derivative of $K$ with respect to $\theta$ quantifies orientation sharpness:

$$
\frac{\partial^2 K}{\partial \theta^2} = -2\kappa K(\theta) + 4\kappa^2 \theta^2 K(\theta)
\quad \Rightarrow \quad \boxed{\kappa = -\frac{1}{K} \cdot \frac{\partial^2 K}{\partial \theta^2} + 2\kappa^2 \theta^2}
$$

At $\theta = 0$, this simplifies to:

$$
\boxed{\left. \frac{\partial^2 K}{\partial \theta^2} \right|_{\theta = 0} = -2\kappa K(0)}
$$

This quantity provides a **metric of angular selectivity**, useful for characterizing the responsiveness of memory projections to orientation changes.

---

### 3. Implications for Clustered Representations

Angular divergence supports memory disentanglement along multiple dimensions:

* Reinforces **semantic orthogonality** between competing attractors
* Encourages **specialization** of projection rays to disjoint regions of memory
* Improves **graceful forgetting** by limiting overlap between non-aligned update flows

This structure can be exploited to construct **multi-angle projection ensembles** with soft or learnable angular basis sets $\{\mathbf{v}_m\}_{m=1}^M$, enabling modular specialization.

> These results extend the stability theorems of Chapter C to directional scenarios - showing that geometric and angular information jointly shape the evolution of distributed memory structures.

> **Note on Forgetting:** While HPM introduces a projection-based update mechanism that is spatially localized and conflict-resilient, **it does not inherently prevent catastrophic forgetting**. If multiple projection rays target overlapping memory regions without explicit replay or structural separation, older memories may be gradually overwritten. To preserve long-term information, the system may require **reactivation of past projections**, **suppressive update schemes**, or **explicit memory partitioning**. HPM provides the *tools* for continual learning — but **stability must be architected, not assumed**.

---

## D.8 Practical Implementation Notes

This section provides engineering-level guidance for implementing the directional projection protocol in practice. While the preceding sections formalize the mathematical and cognitive foundations of HPM, deployment on real systems - especially GPU-based neural backends - demands specific optimization strategies.

---

### 1. Direction Quantization via Learnable Codebooks

To reduce redundancy and improve computational efficiency, directional vectors $\mathbf{v}$ can be drawn from a **learnable codebook**:

$$
\{ \mathbf{v}_m \}_{m=1}^{M}, \quad \mathbf{v}_m \in \mathbb{R}^N, \quad \|\mathbf{v}_m\|_2 = 1
$$

Typical values: $M = 16 \dots 64$. Each projection ray selects a direction via:

* Hard assignment (argmax over similarity to preferred orientation)
* Soft assignment via dot-product attention
* Learned routing based on $u$ or $\Phi(u)$

This enables angular reuse, sparsity, and modular specialization.

---

### 2. Efficient Integration via Beam-Aligned Convolution

Approximate the projection integral:

$$
T(u) = \int W(x) \cdot K(x, \ell_u) \, dx
$$

using **beam-aligned convolution** or **1D FFT** along the direction $\mathbf{v}$:

* Discretize $t \in [t_{\min}, t_{\max}]$
* Extract samples $x_t = \Phi(u) + t \cdot \mathbf{v}$
* Convolve along $t$ using Gaussian weights

This reduces 3D memory traversal to 1D filtering, highly efficient on GPUs.

---

### 3. Stable Direction Vector Normalization

To ensure projection direction is valid during learning:

```python
v = v / max(v.norm(), eps)              # Normalize with epsilon safeguard
loss += lambda_norm * (1 - v.norm())**2 # Optional L2 norm regularization
```

This prevents vanishing or exploding projection rays.

---

### 4. Batched Ray Evaluation

Group rays by common direction $\mathbf{v}_m$. For each batch:

* Precompute $t_x$, $d_\perp$, and kernels $K(x, \ell_u)$
* Use vectorized tensor ops for sampling and projection
* Align projection surface $\mathcal{P}$ to memory grid for fast indexing

---

### 5. Local Update Normalization

When applying local update rules:

$$
W(x) \leftarrow W(x) + \alpha \cdot \frac{\delta(u) \cdot K(x, \ell_u)}{\int K(x, \ell_u) dx}
$$

Normalize the update to prevent over-amplification in sparse kernels.
This ensures graceful adaptation and compatibility with continual learning.

---

### 6. Parameter Scheduling Guidelines

| Parameter               | Effect                          | Tuning Strategy                 |
| ----------------------- | ------------------------------- | ------------------------------- |
| $\sigma$                | lateral resolution              | Set by memory density           |
| $\tau$                  | depth/context scale             | Adaptive or fixed per task      |
| $\alpha$                | update rate                     | Decay over time or confidence   |
| $\lambda_{\text{norm}}$ | directional norm regularization | Small ($10^{-3} \dots 10^{-2}$) |
| $M$                     | number of directional bases     | 16–64 depending on task         |

---

> These implementation notes transform directional HPM from theory to practice - ensuring it remains computationally tractable, stable under learning dynamics, and deployable in large-scale memory-augmented architectures.
