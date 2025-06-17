# Experiment 01: Baseline Convergence of HPM with Exact Projection Geometry

**Related source code**  

The source code of the experiment is: /examples/ex01_minimal_hpm_read_write.py

**Objective**

To establish a baseline for convergence and numerical behavior of Holographic Projection Memory (HPM) when trained with exact, ground-truth projection origins and directions. This serves as a reference experiment demonstrating ideal delta-based learning conditions without uncertainty, validating the integrity of the projection and memory update mechanisms.

---

## Experimental Setup

* **Memory Configuration**: A 3D memory field of shape $64 \times 64 \times 64$ with $C=16$ semantic channels.
* **Projection Count**: $B = 2048$ rays per iteration.
* **Projection Geometry**:

  * Ray origins and directions are sampled randomly but fixed throughout training.
  * Each ray is used consistently to project into memory and to measure error.
* **Target Values**:

  * Fixed random target vectors $T^*(u)$ are assigned to each ray.
  * These targets remain unchanged throughout the experiment.
* **Memory Initialization**:

  * The memory field $W(x)$ is initialized to zeros.
* **Read & Write Mechanism**:

  * At each step, the memory is read at the known origin and direction of each ray.
  * The resulting projection $T(u)$ is compared to $T^*(u)$, and the delta is computed.
  * Memory is updated with:
    $W(x) \leftarrow W(x) + \alpha \cdot (T^*(u) - T(u)) \cdot K(x, \ell_u)$
  * All updates are applied in parallel at each iteration.
* **Optimization Loop**:

  * The procedure is repeated for 50 steps.
  * MSE is recorded after each step.

---

## Observed Results

* Initial MSE: ~0.989
* Final MSE after 50 steps: ~0.00035
* Extremely rapid and smooth convergence.
* No oscillations, divergence, or instability.
* Memory field converges to highly accurate projections.

---

## Interpretation

1. **Validation of Core Mechanism**:

   * With perfect knowledge of ray geometry, HPM demonstrates ideal learning behavior.
   * Projection and update operators behave in a numerically stable and geometrically coherent manner.

2. **High-Efficiency Delta Updates**:

   * The system reaches nearly perfect reconstruction of 2048 target vectors with only local, kernel-weighted updates.
   * Full backpropagation is unnecessary — confirming theoretical predictions.

3. **No Semantic Drift or Conflict**:

   * With fixed geometry, the memory field reorganizes smoothly, forming no divergent regions.
   * Updates remain spatially coherent and directionally focused.

4. **Reference Benchmark**:

   * This experiment sets a definitive baseline for all other learning scenarios.
   * It establishes the lowest empirically observed MSE under ideal projection conditions.

---

## Conclusion

Experiment 01 confirms that HPM functions with high precision and efficiency under idealized conditions, validating both the memory model and the Delta-Learning update mechanism. The rapid, monotonic convergence to near-zero error supports the robustness of projection-based updates and establishes a high-fidelity reference point for evaluating future experiments involving noise, conflict, or approximate geometry.
