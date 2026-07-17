# Continuous mosaic--Ewald coating strategy

Status: proposed implementation strategy. This document does not change the result measure, rod
contract, accepted mosaic measure, or post-integration fitting branch convention.

## Outcome

For each incident ray, construct one continuous Ewald-surface intensity measure containing every
eligible reciprocal family. Sample that measure without an Ewald image, reciprocal-space voxel
field, or retained ray-by-rod-by-orientation product.

The complete design contains:

- the rotated \(m=0\) reciprocal line as a cap-like component;
- both analytic roots of every \(m\ne0\) radial family as ring-like components;
- exact rod-level ordered or stacking strength at the selected \(L\);
- one normalized mosaic measure and one coarea factor; and
- the existing once-only detector mass and deposition path.

Here, coating means a probability/intensity measure on the Ewald surface. It does not mean
materializing a surface matrix.

## Identities

Every physical \((h,k)\) rod retains its own rod identity. For a hexagonal cell, rods also carry
the exact family key

\[
m=h^2+hk+k^2.
\]

The current pipeline assigns these internal intersection labels:

| Support | Retained analytic root | Intersection label |
|---|---|---:|
| \(m=0\) | non-direct root | 0 |
| \(m\ne0\) | lower-\(L\) root | 1 |
| \(m\ne0\) | upper-\(L\) root | 2 |

Labels 1 and 2 are ordered by \(L\); the lower root need not always be negative. These are event
construction labels from the current pipeline. They are not the fitting branches in
[CONVENTIONS.md](CONVENTIONS.md), which use signed reciprocal azimuth labels 0 and 1 and represent
\(00L\) as branchless with a collapsed status. The two identity systems must remain separate.

## 1. Rod strength and family aggregation

For rod \(r=(h,k)\), retain

\[
\mathbf q_r(L)=h\mathbf b_1+k\mathbf b_2+L\mathbf b_3
\]

and evaluate the authoritative event-aligned scattering strength \(S_r(L,\lambda)\). For an
ordered model,

\[
S_r(L,\lambda)=r_e^2\left|F_r(L,\lambda)\right|^2.
\]

Atoms and coherent layers combine as amplitudes inside that ordered model before squaring. A
transition-stacking model instead returns its declared finite-sequence or ensemble intensity; it
need not equal the square of one effective complex amplitude.

Distinct rods are independent channels. Never replace their contribution by
\(\left|\sum_rF_r\right|^2\). For the exact family

\[
\mathcal R_m=\{r=(h,k):h^2+hk+k^2=m\},
\]

define the family scattering-strength profile

\[
S_m(L,\lambda)=\sum_{r\in\mathcal R_m}S_r(L,\lambda).
\]

The catalog already enumerates every physical rod, so each rod enters once. Do not multiply those
rows by symmetry-orbit multiplicity again. Phase and parent populations remain separate
incoherent factors in the once-only mass ledger. Any future nonuniform domain or rod population
requires an explicit contract.

\(S_m\) is an intensity profile, not a complex family structure factor. The implementation keeps
the component \(S_r\) values and rod identities so it can label a sampled event conditionally.

The commonly discussed 121 rods arise only from inclusive bounds
\(h,k\in[-5,5]\), giving \(11\times11\) explicit rods. The algorithm must use whatever complete
rod bounds the detector-support calculation requires.

### Exact condition for a family shortcut

Let \(z\) denote the accepted continuous orientation coordinates and \(b\) an intersection root.
Before reduction, the family density has the form

\[
\rho_{i,m,b}(z)=
\sum_{r\in\mathcal R_m}
S_r(L_{i,r,b}(z),\lambda_i)K_{i,r,m,b}(z),
\]

where \(K\) contains the remaining geometry, mosaic measure, one coarea factor, validity, and
other factors owned by that stage. The faster expression

\[
\rho_{i,m,b}(z)=
S_m(L_{i,m,b}(z),\lambda_i)K_{i,m,b}(z)
\]

is exact only after a declared measure-preserving reparameterization of the family coordinates and
proof of both

\[
L_{i,r,b}(z)=L_{i,m,b}(z)
\quad\text{and}\quad
K_{i,r,m,b}(z)=K_{i,m,b}(z)
\]

for every rod in the family under the accepted orientation measure. If either equality fails,
retain the sum over rods.

After sampling a family point, its rod label follows

\[
P(r\mid z,m,b)=
\frac{
S_r(L_{i,r,b}(z),\lambda_i)K_{i,r,m,b}(z)
}{
\rho_{i,m,b}(z)
}.
\]

When the common-root and common-kernel proof passes, this reduces to \(S_r(L)/S_m(L)\).

## 2. Reciprocal cylinders and the accepted mosaic measure

Define

\[
u=\lVert\mathbf b_3\rVert L,
\qquad
\widehat{\mathbf c}^{\,*}=
\frac{\mathbf b_3}{\lVert\mathbf b_3\rVert},
\qquad
\mathbf q_{0,r}=h\mathbf b_1+k\mathbf b_2.
\]

Using \(u\) matches the existing analytic root measure. An implementation integrating in \(L\)
must apply \(du=\lVert\mathbf b_3\rVert\,dL\) exactly once.

Geometrically, all unrotated rods with the same \(m\ne0\) lie on the cylinder

\[
\lVert\mathbf q_\parallel\rVert=Q_{r,m},
\]

with strength varying along \(u\). The \(m=0\) support is the line

\[
\mathbf q_0(u)=u\widehat{\mathbf c}^{\,*}.
\]

That geometric description does not authorize a new independent cylinder azimuth. The accepted
orientation model has only its existing folded tilt coordinate \(\alpha\) and uniform angular
coordinate \(\beta\), with the same rotation applied to both the in-plane rod offset and rod
direction:

\[
R(\alpha,\beta)=R_{\widehat{\mathbf c}^{\,*}}(\beta)R_{\mathrm{tilt}}(\alpha),
\]

\[
\mathbf q_r(\alpha,\beta,u)=
C\,R(\alpha,\beta)
\left(\mathbf q_{0,r}+u\widehat{\mathbf c}^{\,*}\right),
\]

where \(C\) is the fixed crystal-to-sample rotation.

The normalized orientation measure is

\[
dP_M(\alpha,\beta)
=dP_\alpha(\alpha)\frac{d\beta}{2\pi}.
\]

For the current continuous folded model,

\[
dP_\alpha(\alpha)=2w(\alpha)\,d\alpha,
\]

plus a separate zero-tilt atom when configured. Do not add another \(\sin\alpha\). If this measure
is rewritten as a density with respect to solid angle, transform the density and area element
together.

At zero tilt, varying the existing \(\beta\) traces the raw family cylinder. At nonzero tilt,
\(\beta\) also preserves the accepted correlation between the tilted rod offset and tilted rod
axis. Adding an independent cylinder angle \(\psi\) would generally change that correlation and
the physics.

The proposed family-cylinder optimization is therefore a quotient of this exact per-rod
\((\alpha,\beta)\) pushforward. It may analytically eliminate or reparameterize the existing
uniform angle only after proving equality of the resulting measure. It must not add a second
uniform angle.

For a reciprocal subset \(A\), the authoritative family pushforward is

\[
B_m(A)=
\sum_{r\in\mathcal R_m}
\int dP_M(\alpha,\beta)
\int_{-\infty}^{\infty}
S_r(L(u),\lambda)
\mathbf 1[
\mathbf q_r(\alpha,\beta,u)\in A
]\,du.
\]

The full mosaic reciprocal measure is the incoherent sum

\[
B(A)=B_0(A)+\sum_{m>0}B_m(A).
\]

The \(m\ne0\) terms are ring-like; the \(m=0\) term is cap-like. All use the same mosaic law once.

## 3. Ewald conditioning and analytic roots

Each incident state has its own real internal phase vector \(\mathbf k_i\) and Ewald surface

\[
\mathcal E_i=
\{
\mathbf q:
\lVert\mathbf k_i+\mathbf q\rVert
=
\lVert\mathbf k_i\rVert
\}.
\]

For one rod and orientation, write

\[
\mathbf q(u)=\mathbf q_0+u\widehat{\mathbf d},
\qquad
g_i(u)=
\lVert\mathbf k_i+\mathbf q(u)\rVert
-\lVert\mathbf k_i\rVert,
\]

where \(\mathbf q_0\) and \(\widehat{\mathbf d}\) are produced by the same accepted rotation.
The Ewald coating is the delta-conditioned, or equivalently declared finite-shell-limit, measure

\[
\mu_{i,r}(A)=
\int dP_M
\int S_r(L(u),\lambda_i)
\delta(g_i(u))
\chi_i(\mathbf q(u))
\mathbf 1[\mathbf q(u)\in A]\,du.
\]

\(\chi_i\) is the declared physical support and validity mask. It is not an arbitrary numerical
epsilon. This conditioning, rather than ordinary set restriction to a zero-volume surface, defines
the coating measure.

Squaring the elastic equation gives

\[
u^2+2au+c=0,
\]

with

\[
a=\widehat{\mathbf d}\cdot(\mathbf k_i+\mathbf q_0),
\qquad
c=\lVert\mathbf q_0\rVert^2+2\mathbf k_i\cdot\mathbf q_0.
\]

Let \(D=a^2-c\). Then \(D<0\) has no root, \(D=0\) is tangent, and \(D>0\) has

\[
u_\pm=-a\pm\sqrt D.
\]

For \(m\ne0\), convert each regular root to the exact

\[
L_b=\frac{u_b}{\lVert\mathbf b_3\rVert},
\]

evaluate the strength at that \(L_b\), and label the lower-\(L\) and upper-\(L\) roots 1 and 2.
Define \(\mu_{i,m,b}\) by retaining only the corresponding root. Then

\[
\mu_{i,m}=\mu_{i,m,1}+\mu_{i,m,2}.
\]

Using the existing unsquared constraint, the one root coarea factor is

\[
J_b=
\frac{1}{\left|\partial g_i/\partial u\right|_{u_b}}
=
\frac{\lVert\mathbf k_i\rVert}
{\left|\widehat{\mathbf d}\cdot\mathbf k_{f,b}\right|},
\qquad
\mathbf k_{f,b}=\mathbf k_i+\mathbf q(u_b).
\]

This is a change of measure caused by Ewald conditioning, not an empirical intensity correction.

### The \(m=0\) component

For \(m=0\), \(\mathbf q_0=0\). The two algebraic roots are

\[
u=0,
\qquad
u_*=-2\mathbf k_i\cdot\widehat{\mathbf d}.
\]

The exact direct root is excluded. The retained intersection-0 event is

\[
L_*=\frac{u_*}{\lVert\mathbf b_3\rVert},
\qquad
\mathbf q_*=u_*\widehat{\mathbf d},
\qquad
\mathbf k_f=\mathbf k_i+\mathbf q_*,
\]

with

\[
J_0=
\frac{\lVert\mathbf k_i\rVert}
{\left|\mathbf k_i\cdot\widehat{\mathbf d}\right|}.
\]

Point exclusion alone is insufficient for a continuous \(m=0\) coating. If a smooth mosaic law
crosses \(\mathbf k_i\cdot\widehat{\mathbf d}=0\) while \(S_0(0)>0\), the integral is
logarithmically divergent near the direct root. A normalizable production component therefore
requires one declared physical regularization:

- a finite direct-beam or beamstop neighborhood in \(\chi_i\);
- a detector support gap that excludes that neighborhood; or
- an explicit finite instrument-resolution model.

Do not use an arbitrary epsilon. A zero-width mosaic atom is also a discrete component and must be
handled separately. Kinematic \(00L\) coating remains separate from pure Parratt reflectivity
unless a named composite model explicitly combines them.

### One-coarea rule

Either:

1. form the reciprocal pushforward and condition it on the Ewald surface using the same delta or
   finite-shell measure; or
2. stay in latent orientation and axial coordinates, solve the root, and apply \(J_b\).

Do not apply both. The latent-root form is the recommended implementation because it does not
construct a reciprocal volume or Ewald matrix.

## 4. Continuous sampler

For incident state \(i\), define component \(c=(m,b)\), with \(b=1,2\) for \(m\ne0\) and,
after physical direct-beam support is declared, \(b=0\) for \(m=0\). Let \(z\) be the accepted
nonredundant orientation coordinate and let \(dP_M(z)\) contain the mosaic measure exactly once.

The general detector-conditioned component measure is

\[
d\mu^{\mathrm{cand}}_{i,m,b}(z)=
\sum_{r\in\mathcal R_m}
S_r(L_{i,r,b}(z),\lambda_i)
J_{i,r,b}(z)
W^{\mathrm{once}}_{i,r,b}(z)
dP_M(z).
\]

The indicator for a valid regular root and the physical support mask are included in
\(W^{\mathrm{once}}\). When the common-root and common-kernel proof passes, this reduces to

\[
d\mu^{\mathrm{cand}}_{i,m,b}(z)=
S_m(L_{i,m,b}(z),\lambda_i)
J_{i,m,b}(z)
W^{\mathrm{once}}_{i,m,b}(z)
dP_M(z).
\]

Every downstream factor that changes selection probability, including exit transport and detector
validity, must be evaluated or reused while preparing this measure. It cannot be postponed until
after an unweighted draw.

Define

\[
M_{i,m,b}=
\int d\mu^{\mathrm{cand}}_{i,m,b},
\qquad
T_i=\sum_{m,b}M_{i,m,b}.
\]

Sampling proceeds as follows:

1. Adaptively integrate the component measures and build continuous marginal and conditional
   cumulative distribution functions, or use a proven rejection sampler.
2. Draw \((m,b)\) with probability \(M_{i,m,b}/T_i\).
3. Draw \(z\) from the normalized continuous component. Quadrature nodes normalize the
   distribution; samples are not snapped to those nodes.
4. Recompute the exact root and \(L\), evaluate the retained rod contributions, and draw the
   conditional rod identity.
5. Recompute or retrieve that event's exact \(\mathbf q\), \(\mathbf k_f\), exit transport,
   detector hit, and once-only factors.
6. Emit the event through the existing event, selection, and detector contracts.

Sampling the total measure first and then its conditional labels is mathematically identical.
Whether the hierarchical form is faster is a benchmark question. Structure strengths may be
evaluated as one vectorized exact-\(L\) family batch. An internal interpolant is allowed only after
bounding its error in the detector observable; no event may be silently quantized to a permanent
\(L\) grid.

## 5. Detector measure

The sampler feeds the unchanged complete-pool ledger in
[RESULT_MEASURE.md](RESULT_MEASURE.md). Source mass, phase/parent population, scattering strength,
mosaic/coarea mass, optics, attenuation, footprint, polarization, and detector validity each
enter once before selection; conservative deposition follows selection. Pixel solid angle remains
metadata, and no sampled factor or selection probability is reapplied after the draw.

The current result is scattering mass in \(\mathrm{\mathring A}^2\) per detector-native pixel. If
\(N\) events are drawn from a complete pool of mass \(T_i\), each carries \(T_i/N\). Literal
unit-count photons additionally require calibrated fluence/exposure, sample amount, detector
efficiency, and a stochastic total-count model.

## 6. Performance and parallel execution

The expected implementation fits the existing ray-block and bounded-candidate execution boundary:

- parallelize over independent incident-ray blocks and bounded component tiles;
- keep canonical physical order and key randomness to physical identity rather than worker order;
- compile family metadata, reciprocal geometry, and immutable instrument state once;
- reuse root geometry and adaptive nodes when only structure strengths change;
- rebuild only affected strength values, component masses, and cumulative distributions;
- keep a robust scalar path for tangent or numerically uncertain cases; and
- benchmark serial tiled CPU execution before introducing a fused CPU or GPU kernel.

This compatibility and the speed of the family quotient are hypotheses until measured. The
optimized path must beat equivalent work without changing identities, selected events, mass, or
the detector observable.

## 7. Validation and lightweight handoff

The initial implementation and validation scope is \(m\ne0\) only, covering intersection roots 1
and 2. Compare the continuous sampler with the accepted scalar enumerator for:

- total reciprocal and detector mass;
- per-family and per-intersection-root mass;
- conditional rod fractions;
- exact strength at sampled \(L\);
- Ewald residual and detector coordinates; and
- convergence of detector moments and one small seeded image.

Do not reopen the established cap/ring shape and intensity-conservation proof. The \(m=0\)
equations remain part of the complete strategy, but production activation is deferred until its
physical direct-beam support is declared. It is not part of the initial \(m\ne0\) validation task.

The family-cylinder quotient is accepted only if it reproduces the current per-rod
\((\alpha,\beta)\) measure. If it does not, retain the continuous per-rod sum; correctness outranks
the shortcut.

Use the existing scalar enumerator as the comparison path rather than creating another permanent
coating implementation. Broad sweeps, full images, profiling output, temporary samplers, and
exploratory tests are disposable proof artifacts. Retain only a compact test protecting a unique
scientific invariant or public contract.

At handoff, remove temporary tests, grids, diagnostics, generated data, and superseded code. The
optimized path must reproduce the declared observable within the frozen tolerance and report
equivalent work, wall time, and peak memory.

The authoritative surrounding contracts are [RESULT_MEASURE.md](RESULT_MEASURE.md),
[CONTRACTS.md](CONTRACTS.md), [DOVETAIL_MATRIX.md](DOVETAIL_MATRIX.md), and
[VALIDATION.md](VALIDATION.md). Parallel execution must remain compatible with
[the existing simulation plan](../tasks/parallel_simulation_geometry_fitting_plan.md). The current
orientation measure and root equations remain authoritative in
src/rasim_next/sampling/mosaic.py and src/rasim_next/reciprocal/ewald.py.
