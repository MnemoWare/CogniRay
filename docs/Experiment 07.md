# Experiment 07: Empirical Profiling of Angular Selectivity under Varying Memory Density

## Objective

To empirically quantify the angular selectivity of projection kernels in HPM by measuring the variation in projection responses $T(u)$ under controlled angular perturbations. The experiment isolates the effect of kernel geometry and memory field density on the system’s ability to distinguish closely aligned rays.

---

## Methodology

### 1. Memory Initialization

Create three separate 3D memory fields $W_i(x) \in \mathbb{R}^{64 \times 64 \times 64 \times 16}$, each initialized to zero:

* **Field A**: imprint 512 random rays
* **Field B**: imprint 1024 random rays
* **Field C**: imprint 2048 random rays

Each ray is associated with a unique random target $T^* \in \mathbb{R}^{16}$ and written into the memory using delta-based projection update:

$$
W(x) \leftarrow W(x) + \alpha \cdot (T^* - T(u)) \cdot K(x, \ell_u), \quad \alpha = 0.005
$$

### 2. Angular Perturbation Sweep

For each written ray $\ell_u$ in a given field:

* Fix the anchor point $\Phi(u)$ and original direction $\mathbf{v}_u$
* Define perturbed rays $\ell_\theta$ by rotating $\mathbf{v}_u$ by angle $\theta \in [0^\circ, 30^\circ]$ in 1° increments
* For each $\theta$, compute:

$$
\text{sim}_\theta = \cos\angle(T(0), T(\theta)), \qquad \text{err}_\theta = \|T(0) - T(\theta)\|_2
$$

* Repeat across all rays in the field
* Aggregate mean and standard deviation of $\text{sim}*\theta$, $\text{err}*\theta$ across all samples

### 3. Kernel Variants

Repeat the full experiment with three kernel configurations:

* **Isotropic kernel**: standard radial Gaussian $\times$ axial exponential
* **Anisotropic kernel**: elliptical Gaussian (e.g., $\sigma_\text{lat} \ne \sigma_\text{tan}$)
* **Truncated (hard-edge) kernel**: abrupt spatial cutoff in lateral and/or axial profile

---

## Metrics

* **Angular Selectivity Curve**: plot $\text{sim}*\theta$ and $\text{err}*\theta$ versus $\theta$
* **Empirical Critical Angle $\theta_c^{\text{emp}}$**: angle at which $\text{sim}_\theta < \eta$ (e.g., $\eta = 0.95$)
* **Sensitivity to Density**: compare $\theta_c^{\text{emp}}$ across Fields A, B, C
* **Kernel Comparison**: evaluate sharpness and stability of selectivity across kernel types

---

## Hypotheses

1. Increasing memory density leads to lower angular selectivity due to interference and overlap.
2. Anisotropic kernels sharpen directional discrimination at the cost of spatial generalization.
3. Truncated kernels yield steeper selectivity drop-offs and cleaner critical thresholds.

---

## Outlook

This experiment provides a minimal yet statistically robust foundation for validating theoretical angular selectivity metrics ($\theta_c$, $\Theta(u; \eta)$). Results will guide kernel parameter tuning and geometric design of HPM memory systems for improved addressability, error tolerance, and collision mitigation.

Future extensions include:

* Adaptive kernel scaling based on local density
* Joint analysis of selectivity and projection capacity
* Integration with conflict resolution strategies
