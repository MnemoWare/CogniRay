# Appendix B - Protective Memory Integration in HPM

*Formal characterization of update mechanisms for stable and non-destructive memory modification in projection-based systems.*

---

## B.1 Overview

The Holographic Projection Memory (HPM) framework permits localized memory modification through projection-aligned update rules. Unlike traditional models where parameter updates are global and unconstrained, HPM updates are geometrically guided and spatially bounded, governed by a kernel $K(x, \ell)$ associated with a projection ray $\ell$.

Given a projection mismatch $\delta \in \mathbb{R}^C$ and a geometric path $\ell$ through the memory field, the general form of an update is:

$$
\Delta W(x) = \alpha \cdot \mathcal{F}(x; \delta, W(x), K(x, \ell))
$$

where:

* $W(x) \in \mathbb{R}^C$ is the current memory content at location $x$,
* $\alpha > 0$ is a learning rate or imprinting coefficient,
* $K(x, \ell) \in \mathbb{R}$ is the projection kernel defining locality and directional support,
* $\mathcal{F}$ is a modulation function that encodes the semantic interaction between the update signal $\delta$, current memory state $W(x)$, and geometric locality $K(x, \ell)$.

This appendix formalizes four distinct classes of such update functions $\mathcal{F}$, each corresponding to a different philosophical stance on memory plasticity, stability, and semantic trust:

1. **Unfiltered Delta Update ($\mathcal{F}_\mathrm{delta}$)** - injects the raw projection error weighted by geometry.
2. **Suppressive Amplification ($\mathcal{F}_\mathrm{sup}$)** - favors reinforcement of existing content.
3. **Associative Modulation ($\mathcal{F}_\mathrm{assoc}$)** - balances alignment and local memory density.
4. **Reflexive Resistance ($\mathcal{F}_\mathrm{refl}$)** - enforces high trust and semantic coherence.

Each method defines a unique trajectory through the memory landscape, with increasing degrees of selectivity and resistance to overwriting.

In the following sections, we will mathematically define each modulation function $\mathcal{F}$, examine its structural properties, and compare their behavior with respect to conflict resolution, stability, and compatibility with long-term memory retention.

---

## B.2 Foundations: Projection-Based Memory Update

The core principle of memory modification in Holographic Projection Memory (HPM) systems is based on spatially localized, projection-driven updates. Unlike scalar or token-based addressing schemes, HPM employs geometric rays as the addressing mechanism, which interact with a continuous memory field $W(x)$ through a spatial kernel.

Let $x \in \mathbb{R}^N$ be a point in the memory field, and let a projection ray be defined as a parametric path $\ell(t) = o + t \cdot v$, where $o \in \mathbb{R}^N$ is the ray origin and $v \in \mathbb{R}^N$ is a unit direction vector. The influence of the ray on point $x$ is governed by the projection kernel $K(x, \ell)$, typically constructed as the product of lateral and longitudinal components:

$$
K(x, \ell) = \exp\left(-\frac{d_\perp^2(x, \ell)}{2\sigma^2}\right) \cdot \exp\left(-\frac{|t_x(x, \ell)|}{\tau}\right)
$$


where:

* $d_\perp(x, \ell)$ is the perpendicular distance from $x$ to the ray path,
* $t_x(x, \ell)$ is the axial distance from the ray origin to the closest point on the ray to $x$,
* $\sigma$ controls lateral spread,
* $\tau$ controls axial attenuation.

A projection response is defined by aggregating content from the memory field along the ray:

$$
T = \int_{\mathbb{R}^N} W(x) \cdot K(x, \ell) \, dx
$$

Upon receiving a target response $T^*$ , the projection error is defined as $\delta = T^* - T$. The memory field is then updated to reduce this error by distributing it spatially through the kernel. The update rule is:

$$
\Delta W(x) = \alpha \cdot \mathcal{F}(x; \delta, W(x), K(x, \ell))
$$

The function $\mathcal{F}$ encapsulates how the error $\delta$ is mapped onto the local memory content $W(x)$ under the spatial support of $K(x, \ell)$. In practice, $\mathcal{F}$ can be designed to adaptively modulate the magnitude and direction of the update according to the semantic context, agreement with existing content, and local memory density.

The geometric structure of HPM ensures that updates are inherently local, spatially differentiable, and modulated by projection geometry. This enables adaptive plasticity while maintaining spatial coherence and topological integrity of the memory field.

Subsequent sections examine specific choices for $\mathcal{F}$, each implementing a distinct mode of semantic modulation and memory protection.

---

## B.3 Unfiltered Delta Injection

The most elementary form of memory update in projection-based systems is the direct injection of projection error, weighted solely by spatial proximity. This method applies the full correction vector $\delta \in \mathbb{R}^C$ uniformly across the support of the projection kernel, without modulation by the current memory state or contextual alignment.

The update rule is:

$$
\Delta W(x) = \alpha \cdot \delta \cdot K(x, \ell)
$$

where:

* $\delta = T^* - T$ is the projection error,
* $K(x, \ell)$ is the projection kernel centered along the ray path $\ell$,
* $\alpha > 0$ is a learning coefficient.

This update corresponds to the modulation function:

$$
\mathcal{F}_\mathrm{delta}(x; \delta, W(x), K(x, \ell)) = \delta \cdot K(x, \ell)
$$

This form is agnostic to the existing memory content $W(x)$ and treats all locations within the beam's support equally, with only spatial proximity modulating the injection. It enforces no alignment constraint, no normalization, and no resistance to overwriting.

### Properties:

* **Linearity**: The update is linear in $\delta$ and additive across multiple projections.
* **No semantic filtering**: The content of $W(x)$ does not influence whether or how it is updated.
* **Unconditional plasticity**: All memory locations in the ray path are subject to modification.

### Use Case:

This method serves as a baseline update mechanism and is appropriate for situations where no prior knowledge is stored or preservation of existing memory is not a concern. It is particularly useful in initialization phases or when enforcing hard constraints via projection templates.

### Limitations:

* **Susceptibility to interference**: Since updates are not conditioned on existing content, conflicting updates from intersecting projections will typically overwrite each other — unless all conflicting sources are simultaneously active and geometrically co-projected (i.e., part of the same batch projection), in which case divergence may emerge through self-organized spatial separation.
* **Lack of generalization**: Because there is no filtering, the update does not preferentially reinforce semantically aligned regions.

Despite its simplicity, unfiltered delta injection defines a foundational baseline upon which more selective and protective memory mechanisms are constructed. It establishes the notion of projection-aligned imprinting and provides a reference point for evaluating modulation-enhanced strategies.

---

## B.4 Amplification of Existing Content (Suppressive Write)

The suppressive update mechanism augments the basic delta-based scheme by incorporating the current memory state into the modulation process. This approach preferentially reinforces content that is already represented in memory, while softly attenuating orthogonal or novel components.

The update rule is defined as:

$$
\Delta W(x) = \left( U(x)^2 + R(x)^2 \right)^{1/2} \cdot U(x)
$$

where:

* $U(x) = \alpha \cdot \delta \cdot K(x, \ell)$ is the primary update term,
* $R(x) = \alpha \cdot W(x) \cdot \max K + K(x, \ell) \cdot \text{sign}(W(x))$ is a refresh vector that captures the existing memory orientation,
* $\alpha > 0$ is a learning coefficient.

The modulation function can be expressed as:

$$
\mathcal{F}_\mathrm{sup}(x; \delta, W(x), K(x, \ell)) = \left( U^2 + R^2 \right)^{1/2} \cdot U
$$

This operation has the effect of nonlinearly amplifying components of $\delta$ that align with existing memory content, and dampening contributions that oppose it. The refresh term $R(x)$ injects a context-dependent bias that guides the update trajectory toward reinforcement rather than replacement.

### Properties:

* **Content-sensitive modulation**: The update depends nonlinearly on the current state $W(x)$.
* **Directional amplification**: Aligned components of $\delta$ are emphasized via vector-wise scaling.
* **Smooth consolidation**: Promotes gentle overwriting rather than abrupt replacement.

### Use Case:

Suppressive write is suitable for memory consolidation, reinforcement learning, and scenarios requiring stable retention of existing content. It provides a compromise between plasticity and stability, allowing integration of new information without destructive interference.

### Limitations:

* **Non-symmetric response**: Opposing directions in $\delta$ and $W(x)$ may be suppressed, limiting adaptability in conflicting regimes.
* **Sensitivity to sparsity**: In regions with near-zero memory, the refresh vector may introduce stochastic-like effects.

Suppressive updates introduce an elementary form of semantic alignment by embedding structural memory priors into the update function. This method marks the transition from unconditional plasticity to content-aware modulation, laying the groundwork for more selective mechanisms in subsequent sections.

---

## B.5 Alignment-Guided Update (Associative Write)

The associative update strategy introduces alignment-based modulation to reinforce memory content consistent with existing structure, while suppressing incongruent signals. This is achieved through joint evaluation of directional similarity and local memory magnitude.

The update rule is given by:

$$
\Delta W(x) = A(x) \cdot U'(x)
$$

where:

* $U(x) = \alpha \cdot \delta \cdot K(x, \ell)$ is the raw update vector,
* $R(x) = \alpha \cdot W(x) \cdot \max K + K(x, \ell) \cdot \text{sign}(W(x))$ is a context-dependent refresh term,
* $U'(x) = \left( U(x)^2 + R(x)^2 \right)^{1/2} \cdot U(x)$ is the composite update vector,
* $A(x)$ is a scalar trust coefficient encoding semantic alignment and memory significance.

The alignment measure is defined as:

$$
S(x) = \frac{1}{2} \left(1 + \cos(U'(x), W(x))\right)
$$

and the significance score is:

$$
Z(x) = \frac{\lVert W(x)\rVert}{\max \lVert W(x)\rVert \cdot \alpha}
$$

Thus, the total trust coefficient is:

$$
A(x) = S(x) \cdot Z(x)
$$

leading to the modulation function:

$$
\mathcal{F}_\mathrm{assoc}(x; \delta, W(x), K(x, \ell)) = A(x) \cdot U'(x)
$$

### Properties:

* **Semantic alignment sensitivity**: Updates are encouraged only when consistent with prior content.
* **Magnitude-aware scaling**: Memory regions with high density are considered more reliable.
* **Trust-based gating**: Combines direction and norm to modulate the strength of change.

### Use Case:

Associative write is appropriate for refinement of structured memory, reinforcement of stable patterns, and safe incorporation of new information under contextual consistency. It is particularly suited for high-density memory regions where stability is prioritized.

### Limitations:

* **Suppression of novel input**: Orthogonal or anti-aligned updates are strongly attenuated.
* **Dependence on density normalization**: Poorly scaled memory magnitudes may distort trust computation.

The associative update regime integrates geometric projection, contextual relevance, and internal trust estimation into a unified modulation strategy. It forms the basis for robust memory evolution under continual exposure to related but non-identical inputs.

---

## B.6 Reflexive Resistance (Reflexive Write)

The reflexive memory update mechanism introduces a resistance factor that modulates updates based on both semantic alignment and deviation from trusted internal content. This strategy is designed to safely integrate new information while preserving preexisting content with high trust and semantic coherence.

The update rule is defined as:

$$
\Delta W(x) = \rho(x) \cdot A(x) \cdot U'(x)
$$

where:

* $U(x) = \alpha \cdot \delta \cdot K(x, \ell)$ is the raw delta-based update,
* $R(x) = \alpha \cdot W(x) \cdot \max K + K(x, \ell) \cdot \text{sign}(W(x))$ is the refresh term reflecting memory orientation,
* $U'(x) = \left( U(x)^2 + R(x)^2 \right)^{1/2} \cdot U(x)$ is the combined modulation vector,
* $A(x)$ is a trust-based alignment factor (as defined in the associative update),
* $\rho(x)$ is the **resistance factor**, defined by deviation from the unfiltered update:

$$
\rho(x) = \frac{\left\lVert A(x) \cdot U'(x) - U(x) \right\rVert}{\max_{x} \left\lVert A(x) \cdot U'(x) - U(x) \right\rVert}
$$


This yields the modulation function:

$$
\mathcal{F}_\mathrm{refl}(x; \delta, W(x), K(x, \ell)) = \rho(x) \cdot A(x) \cdot U'(x)
$$

The resistance term $\rho(x)$ quantifies how far the trusted update deviates from the original raw intent. Large deviations signal potential semantic conflict, triggering stronger damping.

### Properties:

* **Adaptive protection**: Memory is only modified when new content aligns and deviates minimally from established representation.
* **Deviation-sensitive damping**: Strongly divergent updates are naturally suppressed in memory regions where existing content conflicts with the proposed change.
* **Long-term consistency**: Core structures remain stable across learning episodes.

### Use Case:

Reflexive write is well-suited for safeguarding memory regions encoding long-term knowledge, identity features, or contextually invariant information. It supports stable incorporation of incremental updates without the risk of catastrophic forgetting.

### Limitations:

* **Conservativeness**: Excessive resistance in sparse or ambiguous regions may inhibit adaptation.
* **Dependence on trust calibration**: Misestimated $A(x)$ may lead to under- or over-protection.

The reflexive memory update strategy fuses geometric locality, semantic alignment, and resistance-based filtering to ensure robust and selective learning. It represents the most restrictive and trust-governed regime in the projection-based memory hierarchy.

---

## B.7 Comparative Summary

This section synthesizes the mathematical and functional properties of the four modulation strategies $\mathcal{F}(x; \delta, W(x), K(x, \ell))$ defined in this appendix. Each update rule occupies a distinct position in the design space of projection-based memory systems, characterized by its sensitivity to content, alignment, and resistance to destructive interference.

We summarize the four regimes below:

| Modulation Function          | Update Rule                            | Memory Awareness | Semantic Filtering      | Resistance / Trust     | Use Context                        |
| ---------------------------- | -------------------------------------- | ---------------- | ----------------------- | ---------------------- | ---------------------------------- |
| $\mathcal{F}_\mathrm{delta}$ | $\alpha \cdot \delta \cdot K(x, \ell)$ | None             | No                      | None                   | Initialization, forced injection   |
| $\mathcal{F}_\mathrm{sup}$   | $\sqrt{U^2 + R^2} \cdot U$             | Weak             | Implicit                | Directional (internal) | Consolidation, mild overwrite      |
| $\mathcal{F}_\mathrm{assoc}$ | $A(x) \cdot U'(x)$                     | Strong           | Explicit (alignment)    | Trust-weighted         | Stable refinement in dense regions |
| $\mathcal{F}_\mathrm{refl}$  | $\rho(x) \cdot A(x) \cdot U'(x)$       | Very strong      | Strict (conflict-aware) | Alignment + resistance | Safe integration under protection  |

### Gradient of Selectivity and Protection

The transition from $\mathcal{F}_\mathrm{delta}$ to $\mathcal{F}_\mathrm{refl}$ represents a progressive shift:

* From raw, unfiltered updates to highly modulated and context-aware adjustments;
* From uniform injection to spatially differentiated response based on alignment and trust;
* From unconditional plasticity to protective stability.

This hierarchy enables memory systems to operate in multiple regimes of learning:

* **Plastic** ($\mathcal{F}_\mathrm{delta}$, $\mathcal{F}_\mathrm{sup}$) — useful during early-stage adaptation or structural initialization;
* **Stable** ($\mathcal{F}_\mathrm{assoc}$, $\mathcal{F}_\mathrm{refl}$) — required for consolidation, safe incremental integration, and long-term memory integrity.

### Formal Implication

Let $\mathcal{F}_i$ and $\mathcal{F}_j$ denote two modulation strategies such that:

$$
\forall x, \lVert\mathcal{F}_i(x)\rVert \geq \lVert\mathcal{F}_j(x)\rVert \implies \mathcal{F}_i \text{ permits more aggressive modification than } \mathcal{F}_j.
$$

This defines a partial ordering over the family $\{ \mathcal{F}_\mathrm{delta}, \mathcal{F}_\mathrm{sup}, \mathcal{F}_\mathrm{assoc}, \mathcal{F}_\mathrm{refl} \}$ by increasing restrictiveness and memory preservation.

In summary, the architecture of $\mathcal{F}$ determines not only the dynamics of learning but also the epistemic boundary between adaptation and preservation. By appropriately selecting or combining these modulation regimes, one can tailor projection-based memory systems for robustness, efficiency, or flexibility across learning domains.
