# Chapter E - Memory Field and Probing

> *This chapter introduces the internal structure of the Holographic Projection Memory (HPM) field and presents a rasterized formulation of projection that is both mathematically principled and computationally tractable. It enables high-throughput memory access via parallelizable, discrete geometric traversal, generalizing naturally to arbitrary memory dimensionality.*

---

Artificial memory systems often rely on discrete keys, flat embeddings, or address-based retrieval. HPM, in contrast, is based on a geometric principle: information is stored across a continuous spatial field, and accessed via directional integration along semantically meaningful rays. In this formulation, memory becomes not an indexable container, but a **reflective medium** - one in which meaning is encountered through structured probing.

Formally, the HPM memory field is defined as a differentiable function $W : \mathbb{R}^N \to \mathbb{R}^C$, mapping coordinates in continuous space to semantic vectors. Each point $x \in \mathbb{R}^N$ encodes latent content - a high-dimensional vector $W(x) \in \mathbb{R}^C$ - that contributes to a projected response when intersected by an external probe. These probes take the form of directed rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}$, where $\Phi(u) \in \mathbb{R}^N$ is the origin of the ray (sampled from a projection surface), and $\mathbf{v} \in \mathbb{R}^N$ is a unit direction vector.

The projected signal at point $u \in \mathbb{R}^{N-1}$ is computed by integrating contributions from the field along the ray $\ell_u$, typically via a Gaussian kernel centered on the ray trajectory. This defines a directional projection operator:

$$
T(u) = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell_u) \, dx
$$

where $K(x, \ell_u)$ is a continuous, localized kernel that weights memory points based on perpendicular distance to the ray and longitudinal attenuation.

While this integral formulation is elegant and differentiable, its direct implementation is computationally expensive: evaluating $K(x, \ell_u)$ for all $x \in \mathbb{R}^N$ is infeasible on real hardware, particularly when $W$ is represented as a large, discrete tensor. Therefore, an efficient approximation strategy is required.

This chapter develops a **rasterized projection framework**, in which rays are discretized, clipped to the memory volume, and traversed using voxel-efficient line-drawing algorithms (e.g., N-dimensional Bresenham). The resulting discrete paths allow projection values to be computed from a sparse neighborhood of relevant memory points. Furthermore, lateral influence - required to simulate beam width - is implemented either via local convolution in memory space or by smoothing over adjacent rays in the projection domain.

This rasterized strategy offers multiple benefits:

* It eliminates the need for ray–voxel intersection tests at runtime
* It permits fully batched and parallelized execution on GPU hardware
* It generalizes naturally to memory fields of dimension $N \geq 2$
* It preserves the geometric structure and locality properties of HPM

> *Although we speak intuitively of shadows or projections, HPM operates via active semantic reflection: rays illuminate latent content, and the response is what returns from the distributed field.*

In the following sections, we formalize the memory field, define its discretization and structure, and develop the complete rasterized projection procedure, including entry–exit clipping, ray traversal, smoothing, and generalization to arbitrary dimensions.

---

### E.1 The Memory Field as a Semantic Medium

The Holographic Projection Memory (HPM) system is built upon a continuous, high-dimensional memory field, denoted as:

$$
W : \mathbb{R}^N \rightarrow \mathbb{R}^C
$$

Here, $x \in \mathbb{R}^N$ represents a spatial coordinate in an $N$-dimensional ambient memory space, and $W(x) \in \mathbb{R}^C$ is a $C$-dimensional latent vector associated with that position. The field $W(x)$ can be interpreted as encoding local semantic content distributed over a differentiable geometric substrate.

---

#### Discrete Representation  

In practice, the continuous field $W(x)$ is realized as a discretized tensor:  

$$
W[x] \in \mathbb{R}^{D_1 \times D_2 \times \dots \times D_N \times C},
$$

where $x \in \mathbb{Z}^N$ indexes the center of a voxel in a regular grid. The spatial resolution $(D_1, \dots, D_N)$ is typically uniform, and the value $W[x]$ represents the semantic vector (or feature embedding) associated with the corresponding voxel.

This formulation maintains full compatibility with the continuous model in the limit as grid resolution increases:

$$
W(x) \approx W[\lfloor x / \delta \rfloor], \quad \text{for small voxel size } \delta > 0.
$$

Interpolation techniques (e.g., trilinear or kernel-based) may be used to recover smooth responses between voxel centers.

---

#### Role of the Channel Space $\mathbb{R}^C$  

Each memory value $W(x) \in \mathbb{R}^C$ is a vector of latent components or channels. The dimensionality $C$ is not fixed by geometry, and is instead chosen based on the expressiveness required by the application:

* $C = 1$: scalar density field (e.g., for volumetric masking)
* $C \gg 1$: high-dimensional latent codes (e.g., for semantic memory or reconstruction tasks)

The interpretation of channels is model-dependent, but generally assumed to be differentiable and continuous. Projection operations defined in later sections (e.g., $T(u)$) are computed independently for each channel and later aggregated or convolved as needed.

---

#### Spatial Reference and Alignment  

The memory field $W$ is embedded in a shared ambient coordinate system $\mathbb{R}^N$, which is also used to define projection rays $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$. All geometric constructs (projection surfaces, rays, attenuation kernels) operate within this space, ensuring full differentiability and compatibility with kernel-based integration schemes.

The origin, orientation, and extent of $W$ are not constrained by the model. Memory may occupy a fixed bounding box or dynamically evolve in form, as long as the probing operators maintain geometric consistency.

---

#### Trainability and Dynamic Update  

The memory tensor $W[x]$ is generally **trainable**, and may be updated:

* **Via standard backpropagation** through projection layers $T(u)$, with gradients $\partial T / \partial W$ given by the projection kernel $K(x, \ell_u)$
* **Locally during inference**, using error-based correction:

$$
W(x) \leftarrow W(x) + \alpha \cdot \delta(u) \cdot K(x, \ell_u),
$$

as introduced in Chapter A.3 and formalized in Chapter D.5. This enables **inference-time plasticity** and selective memory adaptation.

The locality of the projection kernel $K$ ensures that updates only affect regions semantically aligned with the probing ray, thereby preventing widespread corruption of unrelated memory content.

---

#### Interpretability and Modularity  

Because $W$ is defined over a spatial lattice with fixed indexing and structured semantics, its internal organization can be visualized, monitored, and analyzed. This interpretability is essential for:

* Debugging and visualization of memory responses
* Understanding the structure of projection shadows $T(u)$
* Supporting modular and compositional learning paradigms

In summary, the memory field $W(x)$ in HPM acts as a structured, high-capacity, differentiable medium - enabling coherent interaction between spatial geometry and latent semantics. Its design supports both stability under projection and adaptability under learning, forming the backbone of the HPM architecture.

---

### E.2 Projection Rays and Probing Protocol  

Information stored in the HPM memory field is not accessed through discrete indexing or content-based similarity, but rather via structured geometric **probing** - the process of integrating semantic contributions along **directed rays** that traverse the memory volume. These rays originate from a projection surface embedded in the ambient space and are parameterized by coordinates $u \in \mathbb{R}^{N-1}$.

Each such coordinate defines a **ray** $\ell_u(t) \in \mathbb{R}^N$, constructed from a surface mapping $\Phi(u)$ and a direction vector $\mathbf{v}_u$, forming the basis of directional memory access. The response of the memory field to probing is given by the integral of local content $W(x)$ modulated by a kernel $K(x, \ell_u)$ that localizes interaction both **laterally** (via distance to the ray path) and **longitudinally** (via depth attenuation).

This section formalizes the construction of projection rays, introduces conventions for directionality and attenuation, and outlines the operational modes for defining and modulating memory access paths. The probing protocol described here provides the geometric foundation for all downstream projection and update operations in HPM.

---

#### E.2.1 Ray Construction and Notation  

In the Holographic Projection Memory (HPM) framework, access to the memory field $W(x)$ is mediated by directed probing rays that originate from a projection hypersurface and traverse the ambient memory space $\mathbb{R}^N$. Each ray is defined in terms of a projection coordinate $u \in \mathbb{R}^{N-1}$, which indexes a location on the $(N-1)$-dimensional projection surface $\mathcal{P}$.

---

**Ray Parameterization**  

Given a differentiable mapping $\Phi : \mathbb{R}^{N-1} \rightarrow \mathbb{R}^N$, the ray corresponding to coordinate $u$ is defined as:

$$
\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad t \in \mathbb{R},
$$

where:

* $\Phi(u) \in \mathbb{R}^N$ is the origin point of the ray on the projection surface,
* $\mathbf{v}_u \in \mathbb{R}^N$ is a unit-length direction vector associated with $u$,
* $t \in \mathbb{R}$ is the scalar tracing parameter that defines the position along the ray.

The direction vector $\mathbf{v}_u$ may be shared across all $u$ (global direction) or vary locally (per-ray orientation). In either case, normalization is enforced to ensure well-posedness:

$$
\| \mathbf{v}_u \|_2 = 1.
$$

---

**Geometric Role**  

Each ray $\ell_u(t)$ defines a line in $\mathbb{R}^N$ along which contributions from the memory field are aggregated. The ray acts as a **semantic probe**, aligned along direction $\mathbf{v}_u$ and rooted at surface point $\Phi(u)$. Only points in the vicinity of this ray contribute meaningfully to the projection, due to the localized nature of the projection kernel $K(x, \ell_u)$.

---

**Kernel Decomposition**  

The contribution of a memory point $x \in \mathbb{R}^N$ to the projection associated with $\ell_u$ is modulated by a kernel function $K(x, \ell_u)$, defined as a product of two components:

$$
K(x, \ell_u) = K_\perp(d_\perp(x)) \cdot A(t(x)),
$$

where:

* $d_\perp(x)$ is the perpendicular distance from point $x$ to the ray $\ell_u$,

* $K_\perp$ is a transverse weighting kernel, typically Gaussian:

  $$
  K_\perp(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right),
  $$

* $t(x)$ is the axial projection of $x$ onto the ray, i.e., the value of $t$ that minimizes $\| x - \ell_u(t) \|^2$, given by:

  $$
  t(x) = (x - \Phi(u)) \cdot \mathbf{v}_u,
  $$

* $A(t)$ is a longitudinal attenuation function that suppresses distant contributions along the ray.

---

**Exponential Attenuation**  

The default attenuation model is exponential decay with a ray-specific attenuation parameter $\tau_u > 0$:

$$
A(t) = \exp\left(-\frac{t}{\tau_u}\right), \quad t \geq 0.
$$

This form ensures that the kernel $K(x, \ell_u)$ is directionally causal: points further along the ray have progressively reduced influence. The decay rate $\tau_u$ controls the effective receptive depth of the ray and may be:

* Fixed globally,
* Learned per ray coordinate $u$, or
* Adaptively modulated based on external context.

This directional and attenuated formulation of projection rays enables localized, anisotropic sampling of the memory field and forms the basis for the differentiable projection operator $T(u)$ described in subsequent sections.

---

#### E.2.2 Generalizations and Operational Conventions  

The ray construction framework introduced in Section E.2.1 supports a variety of generalizations and implementation conventions that allow for flexible trade-offs between expressiveness, efficiency, and differentiability. These choices influence how rays interact with the memory field and are critical for designing practical HPM systems.

---

**Bidirectional Emission**  

By default, each projection coordinate $u \in \mathbb{R}^{N-1}$ emits a **single ray** $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$, with $t \geq 0$. However, the system can optionally be configured for **bidirectional emission**, where both forward and backward rays are evaluated:

$$
\ell_u^{(+)}(t) = \Phi(u) + t \cdot \mathbf{v}_u, \quad
\ell_u^{(-)}(t) = \Phi(u) - t \cdot \mathbf{v}_u, \quad t \geq 0.
$$

Each ray is processed independently with identical attenuation and kernel structure. The resulting pair of projection values $T^{(+)}(u), T^{(-)}(u)$ can be:

* Summed or averaged: $T(u) = T^{(+)}(u) + T^{(-)}(u)$
* Concatenated: $T(u) = [T^{(+)}(u), T^{(-)}(u)] \in \mathbb{R}^{2C}$
* Used selectively, based on gating or learned preference

**Bidirectional emission is particularly useful when the projection surface lies *inside* the boundary of the memory field.**  

In this configuration, the projection acts as a symmetric bidirectional probe that softly illuminates the memory field from within. The exponential attenuation applied in both directions ensures that the contribution from nearby regions is emphasized, while distant points fade smoothly, avoiding sharp cutoff artifacts.

This results in:

* **Smooth gradient flow** during backpropagation, as there are no abrupt edges or truncation in the integration path.
* **Compact and centered projections** that highlight a local region of the memory field - ideal for introspective decoding or context-dependent memory focusing.
* A geometrically coherent way to implement **soft attention over a spatial neighborhood**, without requiring explicit masks or sampling windows.

This approach is especially powerful when the projection surface is narrow or spatially compressed - it effectively “resonates” within a bounded volume, providing a **focused semantic snapshot** of a localized memory region.


---

**Fixed vs. Learnable Parameters**  

The ray definition depends on two key quantities:

* **Direction vector** $\mathbf{v}_u$
* **Attenuation constant** $\tau_u$

These may be treated as **fixed hyperparameters** or **learnable functions**.

**Fixed configuration:**  

* $\mathbf{v}_u = \mathbf{v}$ for all $u$, with $\mathbf{v} \in \mathbb{R}^N, \|\mathbf{v}\|_2 = 1$
* $\tau_u = \tau \in \mathbb{R}_{>0}$

This mode supports high-performance implementations (e.g., ray caching, rasterized traversal) and simpler batching.

**Learnable configuration:**  

* $\mathbf{v}_u = \mathbf{v}(u)$ or sampled from a learnable codebook: $\mathbf{v}_u = \sum_k a_k(u) \cdot \mathbf{v}_k$
* $\tau_u = \tau(u)$, either directly predicted or derived from local features

Learnable variants enhance flexibility and allow the system to adapt its field-of-view and depth perception based on task demands. See Chapters D.2, D.7, and Q3–Q4 for implementation and gradient recovery strategies.

---

**Convention Flags**  

To simplify experimentation and implementation, a set of **convention flags** is defined to control ray behavior:

* `fixed_v_convention`: Assume all rays share a fixed global direction $\mathbf{v}$
* `fixed_tau_convention`: Use a constant attenuation value $\tau$ for all rays
* `axial_alignment_convention`: Align $\mathbf{v}$ with one of the principal axes (e.g., $x$, $y$, $z$) for memory-efficient traversal

These flags are **not semantic constraints**, but **engineering heuristics** that guide implementation without altering the core geometry.

---

**Modularity vs. Trainability**  

The choice between fixed and adaptive ray parameters reflects a deeper trade-off:

* **Fixed configurations** prioritize efficiency, reproducibility, and clarity. They are well-suited for rasterized projection (see Chapter F) and deployment in constrained environments.
* **Trainable configurations** support task-specific specialization, dynamic focus, and contextual behavior. They align more naturally with gradient-based learning systems.

In practice, hybrid schemes are most likely to be used, where:

* Direction vectors $\mathbf{v}_u$ are drawn from a learned dictionary
* Attenuation $\tau_u$ is predicted from the projection surface $\Phi(u)$
* Only a subset of rays are fully dynamic, while others remain fixed

This modular design ensures that HPM can operate across a spectrum of regimes - from static geometric probing to fully adaptive perceptual routing.

---

### E.3 Entry–Exit Clipping and Ray Activation  

In the context of Holographic Projection Memory (HPM), not all rays are guaranteed to intersect with the memory field. To ensure computational and semantic validity of each projection, rays must be clipped to the valid bounds of the memory region. This section describes the use of geometric intersection logic to define the activation domain of rays and introduces optional attenuation clipping strategies.

---

#### Ray–AABB Intersection  

Let the memory field $W(x)$ be defined over a compact hyperrectangular region (axis-aligned bounding box, or AABB) in $\mathbb{R}^N$. The projection ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ is evaluated only within this region.

The ray is intersected with the AABB to determine valid integration limits:

$$
t_{\text{entry}}, t_{\text{exit}} = \texttt{RayAABBIntersect}(\Phi(u), \mathbf{v}_u)
$$

where:

* $t_{\text{entry}}$: first point of intersection between $\ell_u$ and the memory volume
* $t_{\text{exit}}$: last point of intersection (along $\mathbf{v}_u$)

These values satisfy:

$$
t_{\text{entry}} \leq t_{\text{exit}}, \quad t_{\text{entry}} \geq 0
$$

and can be efficiently computed using branchless slab methods (e.g., inverse direction and bounding box planes).

---

#### Conditional Ray Validity  

Rays that do not intersect the memory region are considered **inactive** and excluded from projection computation. Formally, a ray is active if:

$$
t_{\text{exit}} > t_{\text{entry}} \geq 0
$$

Only active rays contribute to the projection integral:

$$
T(u) = \int_{t_{\text{entry}}}^{t_{\text{exit}}} W(\ell_u(t)) \cdot K(x = \ell_u(t), \ell_u) \, dt
$$

If the ray is inactive (i.e., entirely outside the memory domain), then $T(u) := 0$ or the value is masked from downstream processing.

This conditional gating also supports efficient batching, as inactive rays can be filtered before memory access, reducing overhead.

---

#### Clipped Attenuation (Optional)  

In the standard projection formulation, longitudinal attenuation is given by:

$$
A(t) = \exp\left(-\frac{t}{\tau_u}\right), \quad t \in [0, \infty)
$$

However, when rays are clipped to finite segments $[t_{\text{entry}}, t_{\text{exit}}]$, it is sometimes desirable to **reset** the attenuation to start from $t = 0$ at the entry point. This avoids boundary-induced bias and enforces consistent emphasis near the ray's valid origin.

Define a clipped attenuation function:

$$
A_{\text{clipped}}(t) = \exp\left( -\frac{t - t_{\text{entry}}}{\tau_u} \right), \quad t \in [t_{\text{entry}}, t_{\text{exit}}]
$$

This modification is especially beneficial when $\Phi(u)$ lies outside the memory field and rays enter from its boundary (see also Q10). The clipped variant ensures that all active rays decay smoothly from the actual point of entry, rather than from an external geometric origin.

Whether clipped or standard attenuation is used is an implementation convention and should be documented consistently in training and evaluation pipelines.

---

In summary, entry–exit clipping defines the valid support of each ray within the memory field. It enables selective ray activation, improves computational efficiency, and allows for precise control over attenuation behavior, particularly in cases where the projection surface lies outside the memory volume.

---

### E.4 Discrete Rasterization of Rays  

To enable efficient projection in discretized memory fields, continuous ray integration can be approximated using discrete traversal schemes. This section formalizes the rasterized version of ray sampling, outlines its implementation via voxel stepping, and discusses its compatibility with differentiable learning objectives.

---

#### Voxel-Based Approximation  

Let $W[x] \in \mathbb{R}^{D_1 \times \cdots \times D_N \times C}$ denote the discretized memory field defined over a regular $N$-dimensional voxel grid. Instead of evaluating the continuous projection integral

$$
T(u) = \int_{t_{\text{entry}}}^{t_{\text{exit}}} W(\ell_u(t)) \cdot K(x = \ell_u(t), \ell_u) \, dt,
$$

a discrete approximation traverses a finite sequence of voxels $\{x_i\} \subset \mathbb{Z}^N$ along the ray path and accumulates a weighted sum:

$$
T(u) \approx \sum_{i=1}^{n} W[x_i] \cdot K(x_i, \ell_u) \cdot \Delta t_i.
$$

Here:

* $x_i$: center coordinate of the $i$-th voxel traversed by the ray $\ell_u$
* $\Delta t_i$: approximate step length within voxel $x_i$
* $K(x_i, \ell_u)$: kernel weight at the voxel center, evaluated using the same decomposition as in Section E.2.1:

  $$
  K(x_i, \ell_u) = K_\perp(d_\perp(x_i)) \cdot A(t(x_i))
  $$

---

#### Bresenham-Style Traversal  

The voxel path $\{x_i\}$ is computed using a grid-based traversal algorithm. A common choice is **Bresenham's algorithm** (or its N-dimensional generalizations), which enumerates the set of voxels intersected by a ray in a deterministic, integer-arithmetic fashion.

This rasterization yields:

* Uniform coverage without gaps or overlap
* Fixed step ordering for efficient batching
* Compatibility with memory-coherent access patterns

The initial entry point $x_1$ is determined from the intersection of $\ell_u$ with the voxel grid boundary (see Section E.3). Each subsequent voxel is determined by incrementing the grid index along the direction $\mathbf{v}_u$, using precomputed traversal deltas.

---

#### Approximate Distance and Step Length  

Although Bresenham-style traversal is discrete, the **geometry of the ray is preserved** through analytical reconstruction. For each $x_i$, one may compute:

* Longitudinal projection:

  $$
  t_i = (x_i - \Phi(u)) \cdot \mathbf{v}_u
  $$

* Perpendicular distance:

  $$
  d_\perp(x_i) = \left\| x_i - (\Phi(u) + t_i \cdot \mathbf{v}_u) \right\|_2
  $$

* Step length:

  $$
  \Delta t_i \approx \|x_{i+1} - x_i\|_2, \quad \text{or constant if uniform}
  $$

This preserves compatibility with the continuous projection kernel $K$, while maintaining rasterized efficiency.

---

#### Gradient Compatibility (See Q14)  

Although the voxel stepping path $\{x_i\}$ is inherently non-differentiable, the projection computation can remain differentiable with respect to ray geometry by **reconstructing $t_i$ and $d_\perp$ analytically**.

This allows gradients to be computed for:

* $\Phi(u)$: ray origin (projection surface)
* $\mathbf{v}_u$: ray direction
* $\tau_u$: attenuation coefficient

without requiring differentiable rasterization. The approach is formalized in Chapter F.3 and summarized in Q14.

---

In summary, discrete rasterization of rays enables practical and efficient traversal through a voxelized memory field. When combined with geometric reconstruction of sampling parameters, it preserves full compatibility with gradient-based learning and provides a high-performance alternative to continuous integration.

---

### E.5 Beam Width Strategies (Unified Section)  

The effective spatial influence of each projection ray in HPM is governed not only by its direction and attenuation, but also by its **beam width** - the extent to which nearby memory voxels contribute to the projection. This section outlines alternative strategies for controlling beam width, addressing both lateral and longitudinal integration behavior.

Two primary approaches are considered:

* Surface-oriented convolution, which expands contributions orthogonally to the projection surface.
* Volume-oriented convolution, which integrates over a broader 3D region surrounding the ray path.

These strategies offer complementary trade-offs between resolution, efficiency, and expressiveness, and may be selected based on the degree of angular diversity, hardware constraints, or task requirements.


---

#### E.5.1 Projection-Surface-Based Convolution (Preferred)  

In the projection-surface-based convolution scheme, beam widening is not realized by integrating over an expanded volume around the ray path, but rather through a **post-projection convolution** over neighboring rays on the projection surface. This method is computationally efficient, compatible with rasterized implementations, and preserves gradient flow through well-structured tensor operations.

---

**Formulation**  

Let $T(u) \in \mathbb{R}^C$ denote the projection response associated with a coordinate $u \in \mathbb{R}^{N-1}$ on the projection surface. The convolved output $\widetilde{T}(u)$ is computed as:

$$
\widetilde{T}(u) = \sum_{s \in \mathcal{S}} \omega_s \cdot T(u + s),
$$

where:

* $\mathcal{S} \subset \mathbb{R}^{N-1}$: a finite set of offset vectors (e.g., a local neighborhood around $u$)
* $\omega_s \in \mathbb{R}$: scalar weights forming a convolution kernel (e.g., Gaussian or uniform)
* $T(u + s)$: projection results from rays emitted from nearby surface points

This convolution defines a **lateral beam profile** across the projection surface and produces a smoothed, spatially integrated signal $\widetilde{T}(u)$.

---

**Applicability Constraints**  

Projection-surface-based convolution requires that rays corresponding to neighboring surface points $u + s$ share the **same direction vector** $\mathbf{v}_u$ and attenuation $\tau_u$, or at least vary smoothly across $u$. This ensures that the projections $T(u + s)$ are geometrically aligned and meaningfully composable.

In practice, this strategy is most effective when:

* The projection surface is regular (e.g., a grid)

* The emitted rays form a **parallel bundle**:

  $$
  \mathbf{v}_{u + s} = \mathbf{v}_u, \quad \forall s \in \mathcal{S}
  $$

* The attenuation $\tau_u$ is constant across the neighborhood

Such configurations arise naturally when using `fixed_v_convention` and `fixed_tau_convention` (see Section E.2.2).

---

**Computational and Differentiable Advantages**  

This strategy offers several important benefits:

* **Parallelization**: The convolution is implemented as a surface-wise tensor operation over $T(u)$, enabling vectorized GPU execution
* **Gradient flow**: Gradients backpropagate cleanly through $T(u)$ and into the memory field $W[x]$ via the projection kernel
* **Modularity**: Beam width can be tuned independently of ray geometry, allowing flexible receptive field shaping

Moreover, since the convolution is applied after ray evaluation, it incurs **no additional memory field access**, avoiding redundant sampling.

---

In summary, projection-surface-based convolution provides an efficient and differentiable mechanism for controlling beam width in HPM, particularly when rays form a parallel and uniform bundle. It is the preferred method for implementing smooth lateral integration across a projection surface.

---

#### E.5.2 Memory-Volume-Based Sampling (Alternative)  

In settings where rays exhibit heterogeneous geometry - such as varying direction vectors $\mathbf{v}_u$ or adaptive emission across the projection surface - projection-surface-based convolution becomes inadequate due to misalignment between neighboring rays. In such cases, beam widening may instead be achieved via **direct convolution over memory voxels** in the vicinity of each ray path. This approach trades computational efficiency for geometric precision and is recommended only when lateral coherence across rays cannot be assumed.

---

**Beam Widening via Local Memory Convolution**  

Let $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$ denote the ray associated with coordinate $u$. The memory-volume-based method augments the projection operation by integrating over a local neighborhood around the ray path:

$$
T(u) \approx \int_{t_{\text{entry}}}^{t_{\text{exit}}} \left( \sum_{x \in \mathcal{N}(\ell_u(t))} W(x) \cdot K(x, \ell_u) \right) dt,
$$

where $\mathcal{N}(\ell_u(t))$ is a spatial neighborhood (e.g., 3D window or kernel support) centered on the ray point $\ell_u(t)$.

This approach can be seen as applying a **cross-sectional blur** orthogonal to the ray direction, for each sampled point along the ray trajectory. It results in a smoother, more robust accumulation that accounts for spatial variability around the projected path.

---

**Discrete Implementation**  

In rasterized form, the beam widening is implemented as a convolution over a voxel neighborhood around each ray step $x_i \in \mathbb{Z}^N$:

$$
T(u) \approx \sum_{i=1}^{n} \left( \sum_{\delta \in \mathcal{D}} W[x_i + \delta] \cdot \kappa(\delta) \right) \cdot K(x_i, \ell_u) \cdot \Delta t_i,
$$

where:

* $\mathcal{D} \subset \mathbb{Z}^N$: predefined voxel offsets (e.g., a 3x3x3 cube)
* $\kappa(\delta)$: spatial convolution kernel (e.g., isotropic Gaussian or box filter)
* $K(x_i, \ell_u)$: longitudinal-ray kernel as before

This 2-stage kernel (transverse convolution followed by axial integration) enhances stability when ray directions vary and ensures more complete memory sampling.

---

**Trade-Offs and Use Cases**  

While memory-volume-based sampling provides better geometric fidelity for non-parallel or context-modulated rays, it incurs higher computational and memory overhead:

* **Increased access bandwidth** due to multi-voxel reads per ray step
* **Interpolation required** if memory field $W(x)$ is accessed at non-integer coordinates
* **Limited batching** efficiency due to ray-wise irregularity

It is most appropriate when:

* $\mathbf{v}_u$ varies significantly across $u$
* Beam coherence cannot be assumed
* Projection surface is non-uniform or adaptive

---

In conclusion, memory-volume-based sampling enables accurate beam widening in settings with diverse ray geometries. While less efficient than projection-surface-based convolution, it remains a valuable alternative when geometric coherence cannot be guaranteed.

---

### E.6 Support for Arbitrary Dimensions  

The HPM framework is designed to operate in arbitrary spatial dimensionality, provided that the memory and projection geometries satisfy basic compatibility constraints. All core operations, including projection ray construction, kernel evaluation, and memory interaction, are defined for general $N \geq 2$, with consistent generalization to higher dimensions.

---

#### Memory Field Definition  

Let $N \in \mathbb{N}$, $N \geq 2$. The memory field is defined over an $N$-dimensional voxel grid:

$$
W[x] \in \mathbb{R}^{D_1 \times D_2 \times \cdots \times D_N \times C},
$$

with spatial indexing $x \in \mathbb{Z}^N$ and channel dimension $C \in \mathbb{N}$. This discrete tensor representation corresponds to a continuous function:

$$
W : \mathbb{R}^N \to \mathbb{R}^C,
$$

via interpolation or local smoothing kernels.

---

#### Projection Surface and Ray Geometry  

The projection surface $\mathcal{P}$ is a differentiable $(N-1)$-dimensional manifold embedded in $\mathbb{R}^N$, parameterized by:

$$
\Phi : \mathbb{R}^{N-1} \to \mathbb{R}^N.
$$

Each coordinate $u \in \mathbb{R}^{N-1}$ indexes a location on the surface, and defines a ray $\ell_u(t) = \Phi(u) + t \cdot \mathbf{v}_u$, where $\mathbf{v}_u \in \mathbb{R}^N$ is a unit direction vector. This general formulation applies equally in 2D, 3D, and higher-dimensional cases.

---

#### Dimensional Consistency of Kernels  

All kernels and geometric terms are defined in dimension-independent form:

* Perpendicular distance:

  $$
  d_\perp(x) = \left\| x - \left( \Phi(u) + t(x) \cdot \mathbf{v}_u \right) \right\|_2
  $$

* Axial projection:

  $$
  t(x) = (x - \Phi(u)) \cdot \mathbf{v}_u
  $$

* Composite kernel:

  $$
  K(x, \ell_u) = K_\perp(d_\perp(x)) \cdot A(t(x))
  $$

These expressions involve only Euclidean inner products and norms, and are valid in any $\mathbb{R}^N$ with $N \geq 2$.

---

#### Implementation Considerations  

While the mathematical formulation permits arbitrary $N$, practical implementations are typically constrained to:

* $N = 2$: planar or image-like memory
* $N = 3$: volumetric or spatial memory

These cases benefit from optimized rasterization, data layout, and visualization tools. However, the general structure of HPM remains dimension-agnostic, allowing extensions to higher-order spatial representations (e.g., $N = 4$ for spatiotemporal memory).

To support generic $N$, indexing, kernel computation, and ray traversal routines must be implemented in vectorized or recursive form. These generalizations are described in Chapter F.

---

In conclusion, HPM supports arbitrary spatial dimensionality $N \geq 2$, with a unified mathematical framework and modular implementation structure. Both the projection mechanism and the memory field scale naturally with dimension, preserving semantic coherence and differentiable access across a wide range of geometries.

---

