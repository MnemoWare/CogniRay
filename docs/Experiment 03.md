# Experiment 03: Compatibility of SGD and Delta-Learning in Noisy Projection Conditions

**Related source code**  

The source code of the experiment is: /examples/ex03_minimal_hpm_estimate_read_write_SGD.py

**Objective**

This experiment evaluates the compatibility of classical SGD-based training with Holographic Projection Memory (HPM), and compares its effectiveness to the Delta-Learning mechanism under the same noisy projection regime. It is designed to answer three key questions:

1. Can HPM be integrated with standard gradient-based learning methods?
2. Does Delta-Learning perform comparably to classical SGD when projection geometry is imprecise?
3. Do both strategies converge toward similar accuracy when evaluated under controlled, noise-rich settings?

This experiment builds on the noisy ray recovery protocol introduced in Experiment 02, using it as a common testbed to isolate the effect of learning rule choice.

---

## Experimental Setup

* **Memory Configuration**: A 3D memory field of shape $16 \times 16 \times 16$ (4096 voxels) with $C = 16$ semantic channels.
* **Projection Count**: $B = 4096$ rays per iteration (one per voxel).
* **Target Values**: Fixed random target vectors $T^*(u)$ are generated once and held constant throughout the experiment.
* **Memory Initialization**:

  * The memory field $W(x)$ is initialized with Gaussian noise ($\mu = 0$, $\sigma = 0.1$).
* **Projection Geometry**:

  * At each step, all rays are inferred from the memory field via filtered backprojection and structure tensor analysis.
  * This process is noisy and changes dynamically as $W(x)$ evolves.
* **Learning Rule**:

  * Instead of delta updates, we use PyTorch-style `loss.backward()` and update the memory using:
    $W[x] \leftarrow W[x] - \eta \cdot \text{loss} \cdot \partial \text{loss} / \partial W[x]$
  * The update is thus classical SGD in form, though still guided by projection mismatch.
* **Optimization Loop**:

  * Training runs for 50 steps.
  * MSE between projected values $T(u)$ and target vectors $T^*(u)$ is recorded at each step.

---

## Observed Results

* Initial MSE: ~0.889
* Final MSE after 50 steps: ~0.748
* Stable convergence observed, without divergence or collapse.
* Final accuracy is comparable to Experiment 02 (Delta-Learning): final MSE ~0.714

---

## Interpretation

1. **SGD-Compatible Architecture**:

   * This confirms that HPM is compatible with classical gradient descent and standard autograd-based training loops.
   * Despite the internal use of geometric ray tracing, the memory field $W(x)$ responds well to differentiable loss updates.

2. **Delta-Learning Performs Competitively**:

   * The convergence rate and final error closely match those of Experiment 02.
   * This suggests that Delta-Learning is not a heuristic shortcut, but a viable alternative to SGD — especially in inference-time scenarios where gradient backpropagation is unavailable or undesirable.

3. **Shared Limitation: Ray Inference Accuracy**:

   * Both learning strategies appear to saturate at similar error levels.
   * This supports the hypothesis that the performance ceiling is imposed not by the learning rule, but by the quality of inferred projection geometry.

4. **Projection-Invariant Learning Behavior**:

   * Since the projection mechanism is reused across both experiments, the similar error curves demonstrate that HPM is robust to the choice of local update scheme under fixed projection context.

---

## Conclusion

Experiment 03 confirms that HPM can be seamlessly integrated with standard SGD-style training, while also validating the effectiveness of Delta-Learning under identical conditions. The similar convergence profiles of both methods indicate that the underlying memory dynamics are largely determined by the projection geometry and not the specific update rule. This reinforces the view of HPM as a general-purpose, differentiable memory system — one that supports both classical optimization and alternative learning strategies grounded in geometric feedback.

These results also underscore the potential for hybrid learning approaches: Delta-Learning for fast, local, inference-time updates, and SGD for structured, long-term optimization. Together, they demonstrate the flexibility of HPM as a foundation for continual learning in uncertain environments.
