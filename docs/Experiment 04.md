# Experiment 04: Semantic Drift and Divergence Under Adaptive Projections

**Related source code**  

The source code of the experiment is: /examples/ex04_minimal_hpm_semantic_drift_and_divergence.py  

*Additional:*  
The visualization code for the final memory field state at each phase if: /examples/ex04_visualise_memory_state.py  
The visualization code for the memory dynamics at each phase if: /examples/ex04_visualise_memory_dynamic.py  

**Objective**

To investigate how Holographic Projection Memory (HPM) reorganizes and adapts its internal semantic structure when exposed to sequential, conflicting projection targets. This experiment focuses on whether previously satisfied goals can be reinstated after interference, and how projection geometry responds to shifting semantic constraints. It examines the system's capacity for semantic drift, topological divergence, and adaptive conflict resolution under continuous delta-based updates.

---

## Experimental Setup

* **Memory Configuration**: A 3D memory field of shape $64 \times 64 \times 64$ with $C = 16$ semantic channels.

* **Projection Setup**:

  * Two projection beams, A and B, are defined by fixed (initial) origins and distinct directions.
  * Beam A is held fixed across all stages.
  * Beam B’s target is gradually made orthogonal to A across stages, with its geometry becoming learnable in Stage C.
  * **Importantly, the projection paths of A and B intersect within the memory field** — ensuring that semantic interference and spatial conflict can occur. This overlapping geometry is crucial for eliciting measurable divergence and testing conflict resolution.

* **Target Embeddings**:

  * Beam A is assigned a fixed random target vector $T^*_A$.
  * Beam B receives:

    * A partially mixed variant of $T^*_A$ during Stage A (to avoid early conflict),
    * An orthogonalized vector in Stage B,
    * A new orthogonal vector in Stage C.

* **Update Rule**:

  * Delta-Learning is used throughout:

    $$
    W(x) \leftarrow W(x) + \alpha \cdot (T^*(u) - T(u)) \cdot K(x, \ell_u)
    $$

* **Stage A (Coherent Phase)**:

  * Purpose: jointly imprint A and B when targets are compatible.
  * Geometry: fixed origins and directions.
  * Steps: 50, learning rate $\alpha = 0.005$

* **Stage B (Conflict Phase)**:

  * Purpose: introduce semantic conflict by assigning orthogonal target to B.
  * Geometry: remains fixed.
  * Steps: 50, learning rate $\alpha = 0.005$

* **Stage C (Adaptive Phase)**:

  * Purpose: preserve A while adjusting B’s geometry to accommodate a new orthogonal target.
  * Geometry:

    * A remains fixed.
    * B’s origin and direction become trainable via SGD.
  * SGD Parameters:

    * Learning rate: 0.01
    * Momentum: 0.1
  * Delta-based updates continue for memory.
  * Steps: 100, learning rate $\alpha = 0.0025$

---

## Observations and Interpretation

### Stage A — Coherent Encoding of Compatible Targets

During Stage A, the target vectors for projections A and B are similar ($\delta = 0.25$), and their rays intersect within the memory volume. This results in constructive cooperation, where the memory field develops a shared semantic region capable of supporting both projections simultaneously.

#### Key Observations:

* The field evolves from noise into a **structured semantic zone**, where contributions from both projections reinforce each other.
* The color-coded affinity maps show that **most voxels reside in the blend zone**, indicating dual compatibility.
* However, **a compact difference kernel emerges along the trajectory of beam B**, storing information specific to its slightly divergent target.

Notably, the compact difference kernel associated with projection B is visible even at standard visualization thresholds. Lowering the threshold (e.g., to $10^{-5}$) does not reveal new structure for B, but rather begins to expose **residual traces along the trajectory of projection A**, suggesting that A is increasingly absorbed into the shared representation due to **constructive interference** between A and B, without needing a separate localized zone.

#### Interpretation:

* HPM naturally forms a **shared latent substrate** when projection goals are compatible.
* At the same time, it isolates **goal-specific corrections** into highly localized regions.
* This behavior resembles **residual encoding** — a compressed differential vector field storing deviations without disrupting the primary mode.
* The field self-organizes to **minimize interference while maximizing reuse**, a hallmark of associative generalization.

#### Illustration 1 — Memory Dynamics Over Time:

![Illustration 1 — Memory Dynamics Over Time](files/ex04_stage_a_memory_dynamics.png)

This composite image shows the evolution of the memory field at selected steps (0, 7, 30, 49), highlighting the transition from noise to a clean semantic representation. Notably, it also demonstrates the **gradual assimilation of projection A** into the shared representational field. In this process, we observe how **constructive interference** leads to the compaction of representational structure within the memory space.

#### Illustration 2 — Final State at Different Thresholds:

![Illustration 2 — Final State at Different Thresholds](files/ex04_stage_a_memory_final_state.png)

The left panel uses a threshold of $10^{-3}$, showing the dominant shared semantic region. The right panel, with a lower threshold of $10^{-5}$, reveals **subtle traces along the path of projection A** — evidence of its gradual assimilation into the shared memory structure during Stage A.

---

(Stages B and C to follow.)
