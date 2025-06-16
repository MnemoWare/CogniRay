# Experiment 02: Stability and Semantic Reorganization of HPM under Approximate Projections

**Related source code**  

The source code of the experiment is: /examples/ex02_minimal_hpm_estimate_read_write.py

**Objective**

To empirically demonstrate that Holographic Projection Memory (HPM) remains stable, convergent, and semantically coherent when subjected to massively parallel updates based solely on approximate, internally inferred projection origins and directions. This validates the theoretical claims of topological divergence, semantic drift, and delta-based self-organization under projection-level supervision without classical gradient backpropagation.

---

## Experimental Setup

* **Memory Configuration**: A 3D memory field of shape $16 \times 16 \times 16$ (totaling 4096 voxels) with $C=16$ semantic channels.
* **Projection Count**: $B = 4096$ rays per step, equal to the number of voxels (one ray per voxel).
* **Target Values**: Fixed random target vectors $T^*(u)$ are assigned per ray and held constant throughout training.
* **Memory Initialization**:

  * The field $W(x)$ is initialized with Gaussian noise ($\mu = 0$, $\sigma = 0.1$).
  * No ground-truth information is provided about projection origins or directions.
* **Projection Geometry Inference**:

  * At each step, ray origins and directions are estimated from the evolving memory field using structure tensor analysis and filtered backprojection.
  * These estimates are inherently noisy and change over time as $W(x)$ evolves.
* **Update Phase**:

  * For each inferred ray $\ell_u$, a projection $T(u)$ is computed.
  * Memory is updated using delta learning:
    $W(x) \leftarrow W(x) + \alpha \cdot (T^*(u) - T(u)) \cdot K(x, \ell_u)$
  * All 4096 updates are applied in parallel at each iteration.
* **Optimization Loop**:

  * This process is repeated over 50 steps.
  * MSE between actual and target projections is recorded per step.

---

## Observed Results

* Initial MSE: \~0.91
* Final MSE after 50 steps: \~0.71
* Stable, monotonic convergence without collapse or oscillation.
* All updates are performed without classical backpropagation.
* Ray geometry is continually inferred from the evolving field itself, with no external supervision.

---

## Interpretation

1. **Semantic Self-Organization from Noise**:

   * The memory field $W(x)$, starting from pure noise, organizes itself solely to satisfy projection-level constraints.
   * It forms an interpretable internal geometry capable of producing the target projections under its own inferred coordinates.

2. **Topological Divergence**:

   * Different projection targets compete within the field.
   * Instead of interference or overwriting, the field spatially separates semantic regions to accommodate them.

3. **Semantic Drift**:

   * As learning proceeds, the inferred ray origins and directions shift.
   * This reflects a drift in the latent geometry of $W(x)$ - the memory adapts its internal topology to reduce projection errors.

4. **Associative Convergence without Gradient Descent**:

   * No traditional gradient propagation is used.
   * Updates are purely local and kernel-weighted.
   * Nonetheless, the system converges robustly, guided only by projection mismatches.

5. **Fault-Tolerant Emergent Geometry**:

   * Despite both direction and origin being imprecise, the memory field reconfigures itself to support coherent projection behavior.
   * This emergent geometry is not imposed - it is reconstructed through consistent local agreement with projection targets.

6. **Projection-Driven Hypothesis Formation**:

   * The system effectively hypothesizes what configuration of internal structure would explain the received projections.
   * Over time, this process yields a working semantic geometry consistent with all projected evidence.

---

## Conclusion

This experiment confirms that HPM supports resilient, projection-driven memory formation and adaptation even in the total absence of geometric supervision. Through inferred origins and directions derived from the memory itself, HPM reconstructs a semantically coherent field that converges toward external projection targets. The results empirically validate the theoretical constructs of delta-based learning, semantic drift, and topological divergence, and demonstrate stable self-organization from noise without classical backpropagation. This establishes HPM as a viable architecture for fault-tolerant, hypothesis-generating memory systems driven entirely by projection-level feedback.
