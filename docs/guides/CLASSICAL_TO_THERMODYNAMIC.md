# Classical-to-Thermodynamic Formulation Guide

**Repository:** `thermo-physics`  
**Purpose:** Mathematical and architectural reference for translating digital algorithms (QUBO, Categorical Softmax, Continuous Loss Landscapes) into native thermodynamic Energy-Based Models (EBMs) and Block Gibbs partitions.

---

## Guide 1: Mapping QUBO to Ising SpinNodes

### Mathematical Mapping
In combinatorial optimization, problems are frequently cast as Quadratic Unconstrained Binary Optimization (QUBO):
$$\min_{x \in \{0, 1\}^N} x^T Q x$$

To execute on thermodynamic substrates (spins $s_i \in \{-1, +1\}$), we apply the affine transformation:
$$x_i = \frac{s_i + 1}{2}$$

Substituting this identity into $x^T Q x = \sum_{i=1}^N Q_{ii} x_i + \sum_{i \ne j} Q_{ij} x_i x_j$:

$$x^T Q x = \frac{1}{4} \sum_{i \ne j} Q_{ij} s_i s_j + \frac{1}{2} \sum_{i=1}^N \left( Q_{ii} + \sum_{j \ne i} Q_{ij} \right) s_i + \text{Constant}$$

Equating this with the standard Ising Hamiltonian $E(s) = -\frac{1}{2} s^T J s - h^T s + E_{\text{offset}}$:

$$J_{ij} = -\frac{1}{2} Q_{ij} \quad (i \ne j), \quad J_{ii} = 0$$
$$h_i = -\frac{1}{2} \left( Q_{ii} + \sum_{j \ne i} Q_{ij} \right)$$
$$E_{\text{offset}} = \frac{1}{4} \sum_{i=1}^N \sum_{j=1}^N Q_{ij} + \frac{1}{4} \sum_{i=1}^N Q_{ii}$$

### Local Field & Delta Energy Formulation
When performing single-site or block updates on spin $s_k$, the flip $s_k \to -s_k$ produces an energy change:
$$\Delta E_k = 2 s_k \left(\sum_{j=1}^N J_{kj} s_j + h_k\right) = 2 s_k \cdot \phi_k^{\text{local}}$$

where $\phi_k^{\text{local}}$ is the effective local magnetic field. The Glauber / Metropolis acceptance probability is:
$$P(\text{accept}) = \min\left(1, e^{-\beta \Delta E_k}\right) = \frac{1}{1 + e^{\beta \Delta E_k}}$$

### Penalty Parameter Tuning ($\lambda$)
When encoding equality constraints $g(x) = \sum_i A_{ki} x_i - b_k = 0$, quadratic penalties are added:
$$H_{\text{total}}(x) = H_{\text{cost}}(x) + \lambda \sum_k \left(\sum_i A_{ki} x_i - b_k\right)^2$$

* **If $\lambda$ is too small:** The thermodynamic sampler finds low-energy states that violate feasibility (e.g. invalid tours in TSP or power deficits in SCUC).
* **If $\lambda$ is too large:** The energy landscape becomes an ultra-steep canyon where valid states are separated by towering barrier heights, causing MCMC mixing to freeze ($\tau_{\text{int}} \to \infty$).
* **Golden Rule:** Scale $\lambda$ such that a single constraint violation creates an energy penalty strictly greater than the maximum cost improvement from that violation:
  $$\lambda > \max_{x, \Delta x} \left| \frac{\Delta H_{\text{cost}}}{\Delta g(x)^2} \right|$$

### Failure Modes & Common Pitfalls (Guide 1)
1. **Omission of the 0.5 Factor on Undirected Graphs:**  
   Passing a full symmetric matrix $J$ to an energy evaluator without dividing by 2 doubles the coupling energy, doubling effective $\beta$ and freezing the chain prematurely.
2. **Diagonal Terms in Couplings:**  
   In Ising models, $s_i^2 = (+1)^2 = 1$ is an invariant constant. Placing diagonal terms $J_{ii} \ne 0$ distorts the local field calculation $\Delta E_i = 2 s_i (\sum J_{ij} s_j + h_i)$. Always zero the diagonal: `np.fill_diagonal(J, 0.0)`.
3. **Static Penalty Freezing:**  
   For hard combinatorial constraints, static large $\lambda$ traps sampling in sub-optimal local basins. Use **dynamic penalty ramping** ($\lambda(t): \lambda_{\text{soft}} \to \lambda_{\text{hard}}$) to allow unconstrained exploration before snapping into feasible manifolds.

---

## Guide 2: Converting Categorical Softmax to Categorical EBM Factors

### Mathematical Mapping
In deep learning, discrete choice over $K$ alternatives is parameterized via the Softmax distribution:
$$P(y = k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = \operatorname{softmax}(z)_k$$

In thermodynamic computing, discrete categorical choice is represented by a 1-of-$K$ one-hot Potts spin vector:
$$s \in \{0, 1\}^K, \quad \text{subject to } \sum_{k=1}^K s_k = 1$$

Pairwise correlations between categorical variables $u$ and $v$ are encoded by interaction tensors:
$$E(s_u, s_v) = -\sum_{k=1}^K \sum_{l=1}^K W_{kl}^{(u,v)} s_{u,k} s_{v,l} - \sum_{k=1}^K b_k^{(u)} s_{u,k}$$

### Multi-Color Chromatic Block Gibbs Partitions
Direct sequential sampling of categorical variables induces long pointer-chasing latencies. By building the dual factor graph where nodes are categorical variables and edges are non-zero interactions:
1. Compute the chromatic number $\chi(G)$ via greedy graph coloring.
2. Group all variables of color $c \in \{1, \dots, \chi(G)\}$ into an independent block $\mathcal{B}_c$.
3. Within block $\mathcal{B}_c$, **all categorical variables are conditionally independent given the remaining blocks**.
4. Evaluate all logits $\tilde{z}_{u,k} = b_k^{(u)} + \sum_{v \notin \mathcal{B}_c} W_{k, s_v}^{(u,v)}$ simultaneously across JAX SIMD or parallel hardware cores, and sample categorical states via vectorized Gumbel-max or inverse CDF:
   $$s_u \sim \operatorname{Categorical}\left(\operatorname{softmax}(\beta \tilde{z}_u)\right) \quad \forall u \in \mathcal{B}_c \text{ in parallel}$$

### Failure Modes & Common Pitfalls (Guide 2)
1. **Invalid Chromatic Partitions (Detailed Balance Violation):**  
   If two connected variables are assigned to the same update block, updating them simultaneously violates detailed balance and samples from an unphysical nonequilibrium stationary state.
2. **Exponential Color Explosion:**  
   Dense all-to-all categorical graphs have chromatic number $\chi = N$, collapsing block parallelism back to sequential single-variable updates. Use low-rank factor approximations or auxiliary latent spins to sparsify the graph before coloring.
3. **Log-Sum-Exp Underflow:**  
   When calculating conditional categorical probabilities at large $\beta$, standard $\sum e^{\beta z_k}$ overflows 32-bit floats. Always subtract $\max(z)$ prior to exponentiation.

---

## Guide 3: Continuous Potential Mapping (Langevin / Double-Well vs. Discrete Spins)

### Mathematical Formulation
While Ising models describe discrete two-state systems, physical TSUs can also operate in the continuous regime via stochastic differential equations (Overdamped Langevin Dynamics):
$$d X_t = -\nabla V(X_t) \, dt + \sqrt{2 \beta^{-1}} \, d W_t$$

where $V(X)$ is a continuous potential energy function and $W_t$ is standard Brownian motion.

### Kramers Escape Rate Across Potential Barriers
Consider a symmetric double-well potential:
$$V(x) = \frac{\lambda}{4} (x^2 - a^2)^2 = \frac{\lambda}{4} x^4 - \frac{\lambda a^2}{2} x^2 + \frac{\lambda a^4}{4}$$

* Minima (ground states): $x^* = \pm a$
* Saddle point (barrier): $x = 0$
* Barrier height: $\Delta V = V(0) - V(\pm a) = \frac{\lambda a^4}{4}$

By Kramers' escape theory, the transition rate between the two wells is:
$$r_{\text{Kramers}} = \frac{\sqrt{V''(a) |V''(0)|}}{2\pi} e^{-\beta \Delta V}$$

### Discrete Spin Discretization
A continuous double-well oscillator maps directly to an effective discrete Ising spin:
$$s = \operatorname{sgn}(X_t) \in \{-1, +1\}$$

When multiple oscillators are linearly coupled ($V_{\text{inter}}(X, Y) = -J X Y$):
$$V_{\text{total}}(X) = \sum_i V(X_i) - \frac{1}{2} \sum_{i,j} J_{ij} X_i X_j$$

At low temperature ($\beta \Delta V \gg 1$), intra-well oscillations relax quickly ($\tau_{\text{fast}} \sim 1 / V''(a)$) while inter-well transitions occur on the slow Kramers timescale ($\tau_{\text{slow}} \sim e^{\beta \Delta V}$), rigorously demonstrating the equivalence between continuous Langevin dynamics and discrete Ising spin flips.

### Failure Modes & Common Pitfalls (Guide 3)
1. **Euler-Maruyama Step Size Instability ($dt$):**  
   Choosing $dt > 2 / |V''(x_{\max})|$ causes numerical divergence in polynomial potentials ($x^4$). Always verify the Courant-Friedrichs-Lewy (CFL) condition $dt < 1 / \max |\nabla^2 V|$.
2. **Thermal Noise Bias:**  
   In discrete software approximations, using low-quality PRNGs or insufficient pseudo-random bits introduces periodic correlations that destroy Langevin ergodicity. Physical thermodynamic hardware uses true physical Johnson-Nyquist thermal noise, inherently avoiding PRNG cycle limits.
3. **Aggressive Annealing Schedules (Quenching):**  
   Cooling $\beta(t)$ faster than logarithmic schedule $\beta(t) \sim \ln(1 + t)$ guarantees trapping in local metastable double wells. On rugged landscapes, maintain non-zero thermal noise to allow barrier tunneling.
