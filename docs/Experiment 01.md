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

* Initial MSE: \~0.99
* Final MSE after 50 steps: \~0.067
* Rapid, stable convergence.
* No oscillations, divergence, or instability.
* Memory rapidly aligns with target projections.

---

## Interpretation

1. **Validation of Core Mechanism**:

   * With perfect knowledge of ray geometry, HPM exhibits expected learning behavior.
   * The projection and update operators are numerically stable.

2. **Efficiency of Delta-Based Updates**:

   * The system achieves near-perfect reconstruction of 2048 target vectors using only local kernel-based updates.
   * No global backpropagation is needed.

3. **No Semantic Drift or Divergence**:

   * Since geometry is fixed and accurate, no reorganization or bifurcation of memory occurs.
   * The field adjusts smoothly and uniformly.

4. **Reference Benchmark**:

   * This experiment serves as a control for all further tests.
   * It establishes a lower bound on achievable MSE and an upper bound on convergence speed.

---

## Conclusion

Experiment 01 confirms that HPM functions correctly under idealized conditions with full access to true projection geometry. The memory field aligns rapidly with the desired projections, validating the delta-learning mechanism and local kernel updates. This experiment provides a baseline reference against which all approximate, uncertain, or degraded scenarios (such as in Experiment 02) can be compared.
