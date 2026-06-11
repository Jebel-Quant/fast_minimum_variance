# Referee Report — SIAM Journal on Financial Mathematics (SIFIN)

**Manuscript:** *From Marchenko–Pastur to Woodbury: Direct Solvers for Long-Only
Mean-Variance Portfolios* (Schmelzer, Stoll, Wolf)

**Recommendation: Major revision.**

---

## Summary of the paper

The paper considers the long-only mean-variance problem under the
"constant residual eigenvalue" (CRE) covariance estimator obtained by
Marchenko–Pastur (MP) eigenvalue clipping. Its central observation is that the
CRE estimator has low-rank-plus-identity form
$T_0 = \bar\lambda I + U_k \Delta_k U_k^\top$, so the inner solve of a
primal-dual active-set method can be performed via the Woodbury identity at
$O(n_a k^2 + k^3)$ per outer step instead of assembling and Cholesky-factoring
the $n_a\times n_a$ restricted system at $O(n_a^2 k + n_a^3)$. The eigenpairs are
obtained by randomised SVD applied directly to the return matrix at $O(nTk)$,
and the paper accounts for total pipeline cost over a sweep of $S$ solves.
Benchmarks on synthetic and S&P 500 data report 5×–28× speedups over a
KKT-Cholesky baseline and three to four orders of magnitude over CVXPY/Clarabel,
plus a $k$-sensitivity audit showing the MP threshold is not a sensitive
hyperparameter.

## Overall assessment

The paper is clearly written, the engineering contribution is real and useful,
the experiments are carefully described (hardware, library versions, warm/cold
distinction), and the $k$-sensitivity audit is a genuinely practical idea. The
honest caveat about what the CVXPY timings measure is welcome.

However, the manuscript currently has (i) a mathematical error in the stated
algorithm (the dual-feasibility check uses the gradient of the *wrong*
objective), (ii) a crossover analysis whose headline constant
($k \lesssim 1.618\,n_a$) is an artefact of dropping a term that is *not*
lower-order near the crossover — the exact unit-constant crossover is $k < n_a$,
(iii) a discrepancy between the estimator implemented and the CRE estimator of
the cited literature (trace is not preserved), and (iv) novelty claims that
overlook the classical literature on exploiting factor structure in portfolio
QPs (Perold 1984) and on parametric frontier tracing (critical line algorithm).
None of these is fatal — the empirical conclusions will survive — but all of
them must be fixed before publication, and several statistical claims need to be
tempered or supported. Hence major revision.

---

## Major comments

### M1. The dual-feasibility gradient in Algorithm 1 is for the wrong objective

The QP being solved (Section 2) is
$\min_w\, w^\top T_0^{\mathrm{RMT}} w - \rho\,\hat\mu^\top w$. Its KKT
stationarity condition involves the gradient $2\,T_0^{\mathrm{RMT}} w - \rho\hat\mu$.
Yet the dual step in Section 2 and in Algorithm 1 (line 12) computes

$$g_i = \tfrac{2}{T}(X^\top X w)_i - \rho\hat\mu_i,$$

i.e. the gradient of the *sample-covariance* objective $w^\top \hat\Sigma w$.
These differ by $2\,U_0(\Lambda_0-\bar\lambda I)U_0^\top w \neq 0$. As written,
the algorithm checks dual feasibility for a different QP, so its fixed point is
in general **not** the optimum of the stated problem, and Proposition 2.1
(finite termination at the optimum) does not apply to Algorithm 1 as printed.

The accompanying paragraph ("*KKT gradient requires $X$ … this is the only
point where $X$ enters the solve*") is built on this error. Note that with the
correct gradient the story actually *improves*: $T_0^{\mathrm{RMT}} w$ costs
$O(nk)$ by Proposition 3.1(4), which is cheaper than the $O(nT)$ matvec pair
with $X$, and the return matrix then never enters the solve at all.

I strongly suspect the implementation is correct and the manuscript mis-states
it (the text reads like it was carried over from a companion paper in which the
objective genuinely involves $\hat\Sigma$). Please fix Algorithm 1, the dual
step in Section 2, and the surrounding discussion, and confirm that all
reported timings were produced with the correct gradient. A simple and
convincing check, which I ask the authors to add: report
$\|w_{\mathrm{Woodbury}} - w_{\mathrm{CVXPY}}\|_\infty$ explicitly. The caption
of Table `tab:solvers` currently says only that "all **direct** solvers produce
identical weights", which is consistent with all of them sharing the same
(possibly wrong) outer loop and proves nothing about optimality.

### M2. The crossover constant $1.618\,n_a$ is spurious; the correct statement is $k < n_a$

Remark 4.2 derives the crossover by comparing $n_a k^2 + k^3$ against
$n_a^2 k + n_a^3$ and then "dropping the lower-order terms $k^3$ and $n_a^2$"
to obtain $k \lesssim \tfrac{1+\sqrt5}{2}\, n_a$. But near the crossover one has
$k \sim n_a$, where $k^3$ is exactly the same order as $n_a k^2$ — it is not a
lower-order term there. Keeping all terms, with $x = k/n_a$:

$$n_a k^2 + k^3 < n_a^2 k + n_a^3 \iff x^3 + x^2 - x - 1 < 0 \iff (x+1)^2(x-1) < 0 \iff x < 1.$$

So the exact unit-constant crossover is the much cleaner $k < n_a$. The same
conclusion holds with realistic FLOP constants (SYRK assembly $\sim n_a^2 k$,
Cholesky $\sim n_a^3/3$ vs. $\sim n_a k^2$ and $k^3/3$): the condition
$x^3 + 3x^2 - 3x - 1 < 0$ factors as $(x-1)(x^2+4x+1) < 0$, again giving
$x < 1$. The golden-ratio constant is an artefact and, worse, it *overstates*
the Woodbury region: at $k = 1.2\,n_a$ (inside the claimed region) Woodbury
costs $\approx 3.2\,n_a^3$ versus $\approx 2.2\,n_a^3$ for Cholesky.

Consequences to fix:

- Replace $k \lesssim 1.618\,n_a$ by $k < n_a$ in the abstract, Remark 4.2,
  Remark 4.1, and the conclusions. The empirical conclusion is unchanged
  ($k/n_a \approx 0.48$), and the corrected statement is simpler.
- Remark 4.1 currently says "when $n_a < k$ the approximate crossover condition
  fails" — with the corrected analysis, $n_a = k$ *is* the crossover, which
  makes that remark exact rather than approximate.
- The sentence quantifying the dropped terms ("understates the Woodbury
  advantage by at most 52%…") should be deleted along with the approximation.
- Note that at the benchmark ratio $x \approx 0.48$ the FLOP-count advantage is
  only about $3\times$, while the measured advantage is $5\times$–$28\times$.
  The paper attributes the excess to cache effects (Remark 4.2, Section 7.1).
  That is plausible but currently asserted, not measured — see M6.

### M3. The estimator implemented is not the CRE estimator of the cited literature (trace not preserved)

Section 3 sets $\bar\lambda = \mathrm{tr}(\hat\Sigma)/n$, the mean of **all**
eigenvalues, and calls the result "exactly the CRE estimator studied by Bun,
Bouchaud and Potters". It is not. The standard CRE/clipping prescription
(Laloux et al. 1999; Bun–Bouchaud–Potters 2017) replaces the bulk eigenvalues
by their **bulk average**
$\bar\lambda_{\mathrm{bulk}} = \big(\mathrm{tr}\hat\Sigma - \sum_{j\le k}\lambda_j\big)/(n-k)$,
precisely so that the trace — total variance — is preserved. With the paper's
choice, $\mathrm{tr}(T_0) = \mathrm{tr}(\hat\Sigma) + \sum_{j\le k}(\lambda_j - \bar\lambda) > \mathrm{tr}(\hat\Sigma)$,
and since the leading eigenvalues of an equity covariance carry a large
fraction of total variance, the inflation is material (plausibly tens of
percent at the S&P 500 calibration). This is not a harmless rescaling: changing
$\bar\lambda$ changes the *relative* weight of bulk versus signal directions
and therefore changes the minimum-variance weights.

The authors must either (a) switch to the trace-preserving bulk average — all
structural results (positive definiteness, $\delta_j > 0$, the Woodbury
algebra, the complexity analysis) go through verbatim — or (b) keep their
variant but say explicitly that it differs from the CRE estimator of the cited
references, justify it, and quantify the effect on weights. In either case the
attribution sentence below Proposition 3.1 must be corrected. Relatedly,
calling $\mathrm{tr}(\hat\Sigma)/n$ the "bulk mean eigenvalue" (Section 3.2) is
incorrect terminology; it is the grand mean, which equals the MP variance
estimate $\hat\sigma^2$ — worth stating, since it is the one identity that
makes $\hat\lambda_+ > \bar\lambda$ automatic.

### M4. "Replacing noise eigenvalues with $\bar\lambda$ is the statistically correct operation" is an overclaim

Remark 2.1 asserts that CRE clipping is "not an approximation but the
statistically correct operation". This is contradicted by the very literature
the paper cites: the oracle rotationally-invariant estimator (Ledoit–Péché
overlaps; Bun–Bouchaud–Potters 2017, §6) replaces bulk eigenvalues by values
that are *not constant* across the bulk, and nonlinear shrinkage
(Ledoit–Wolf 2012, 2017, 2020) strictly dominates clipping in Frobenius loss.
Within the RMT school, CRE is regarded as a useful, simple approximation to
RIE — not as optimal. The remark should be rewritten: the legitimate argument
for $\alpha=1$ here is *algebraic* (only $\alpha=1$ preserves the
low-rank-plus-identity structure that enables Woodbury), and the legitimate
defence of CRE is that it is a standard, well-studied estimator whose
statistical properties are documented elsewhere.

Two related points:

1. **Positioning vs. nonlinear shrinkage / RIE is missing entirely.** These
   estimators are full-rank perturbations of the eigenvalue spectrum and are
   *not* Woodbury-compatible — that is an important limitation of the proposed
   pipeline and should be stated openly, with the relevant citations
   (Ledoit–Wolf; Ledoit–Péché; the RIE of Bun et al.). The absence of any
   Ledoit–Wolf reference is conspicuous.
2. **No out-of-sample evidence.** The paper's value proposition is "same
   statistical quality, much faster", but statistical quality is taken on
   faith. Either include a brief out-of-sample experiment (realised volatility
   of the min-variance portfolio under CRE vs. sample covariance vs. a
   shrinkage baseline on the S&P 500 sample, rolling windows) or confine all
   statistical claims to citations. Given that the repository of the companion
   experiments evidently produces such results, including them would
   substantially strengthen the paper.

### M5. Novelty positioning: factor-structure QP solvers and the critical line algorithm

The claim in the introduction that "all existing implementations of CRE
cleaning form $T_0$ as a dense $n\times n$ matrix and pass it to a
general-purpose solver" is too strong (it cannot be verified for proprietary
implementations) and the broader idea — exploiting diagonal-plus-low-rank
covariance structure inside an active-set portfolio QP via
Sherman–Morrison–Woodbury — is classical: Perold (1984, *Management Science*,
"Large-scale portfolio optimization") is the standard reference, and the
technique is folklore in commercial factor-model optimisers. The genuinely new
element here is the *combination*: an RMT-determined $k$, the rSVD
preprocessing with end-to-end pipeline accounting, the crossover analysis, and
the measured comparison. The introduction should say exactly that and cite
Perold.

Separately, the $\rho$-grid frontier sweep of Section 7 re-solves at 50 grid
points, while the critical line algorithm (Markowitz 1956; see also
Niedermayer & Niedermayer 2010 for a fast implementation) traces the exact
piecewise-linear frontier by pivoting from breakpoint to breakpoint — which is
both exact and likely faster than 50 warm-started solves. Since the paper
already cites Markowitz (1956) for the piecewise-linear structure, it should
explain why a grid sweep is used instead of CLA (or compare against it).

### M6. Proposition 5.1 does not follow from the cited theorem, and is false as stated

Proposition 5.1 bounds
$\|\hat\Sigma - \hat U_k\hat\Lambda_k\hat U_k^\top - \bar\lambda(I-\hat U_k\hat U_k^\top)\|_F
 \le (1 + 4\sqrt{k+p}/(p-1))\,\sigma_{k+1}^2/T$,
attributing it to Theorem 1.1 of Halko–Martinsson–Tropp. That theorem bounds
$\mathbb{E}\|A - QQ^\top A\|$ (spectral norm, with an extra
$\sqrt{\min(m,n)}$ factor in the Frobenius/average-case variants, Thms.
10.5–10.6), for the sketched matrix itself — not for the trace-lifted
reconstruction on the left-hand side here. Moreover the left-hand side cannot
be bounded by $\sigma_{k+1}^2/T$ at all: it contains the term
$\bar\lambda(I - \hat U_k\hat U_k^\top) - U_0\Lambda_0U_0^\top$-type residue
whose Frobenius norm is of order $\bar\lambda\sqrt{n-k}$ *regardless of sketch
quality* (numerically $\approx 5\times10^{-3}$ at the benchmark, versus the
claimed bound $\approx 4\times10^{-3}$ — the inequality is not even satisfied
with comfortable margin, and for larger $n$ it fails outright since the LHS
grows like $\sqrt{n}$). Please restate the proposition as a bound on the
quantity rSVD actually controls — e.g. $\|\hat\Sigma_k - \hat U_k \hat\Lambda_k
\hat U_k^\top\|$ or the subspace alignment $\|U_kU_k^\top - \hat U_k\hat
U_k^\top\|$ — with the correct norm and constants, and derive the portfolio
implication from that. The empirical subspace-error and 0.14 bp weight-change
evidence in Section 5 is convincing on its own; the proposition just needs to
say something true.

Also: the first-order argument converting a 1.2% subspace rotation into
"$0.012\times 12\,\mathrm{bp} \approx 0.14$ bp" treats the $k$-sensitivity
result (a *rank-one change* of the subspace) as a Lipschitz bound for
*rotations within* the subspace. These are different perturbation classes; the
heuristic is fine if labelled a heuristic, but the chain "strictly smaller
perturbation" is not an argument. A direct numerical comparison of
$w_{\mathrm{dense}}$ vs $w_{\mathrm{rSVD}}$ is already implied — just report
that number in a table and drop the heuristic.

### M7. Algorithm 1, the prose, the proof, and (apparently) the code all describe different outer loops

Four mutually inconsistent descriptions:

- Section 2 prose (dual step): "add **the most violated** asset" (single
  addition).
- Algorithm 1, line 13: $\mathcal{A} \leftarrow \mathcal{A}\cup\mathcal{V}$ —
  add **all** violated assets.
- Proposition 2.1's proof invokes the *single-exchange* parametric-QP pivot
  argument, which does not cover bulk additions/removals (bulk exchanges are
  exactly the setting where active-set methods can cycle).
- The reference implementation appears to add a single asset per dual step but
  drop negative-weight assets in bulk (with a magnitude-dependent rule), and
  uses the mean rather than the median of the active gradients for small
  active sets.

Please make the printed algorithm match the implementation that produced the
timings, and make the termination proof cover the algorithm as printed. While
doing so: once the gradient of M1 is corrected, all active-set components of
the true gradient are *equal* (to $\hat\lambda$) at the subproblem optimum, so
the median is exact rather than a robustness heuristic — the justification
becomes one line. Finally, $\varepsilon = 10^{-6}$ is an absolute tolerance
applied to gradients whose natural scale is $\bar\lambda \sim 10^{-4}$ for
daily returns; a scale-relative tolerance (or a remark on units) is needed for
the method to be robust across data scalings.

---

## Minor comments

1. **Headline numbers disagree.** Abstract and conclusions: "over $5{,}000\times$
   cold vs CVXPY"; Section 7.1: "over $7{,}000\times$". From Table 3 the sweep
   ratio is $\approx 5{,}300\times$ and the single-solve ratio
   $\approx 7{,}900\times$. Pick one convention and state it in both places.
2. **Preprocessing speedup mismatch.** Section 5 text says "$1.9\times$ faster";
   the table data says $1.8\times$ ($0.0127/0.0069 = 1.84$). Harmonise.
3. **Internal inconsistency on asymptotics.** Remark 4.2 says the Woodbury
   advantage "grows with $n$ because the $O(n_a^3)$ term dominates", but under
   the paper's own scaling model ($k \approx n/21$, $n_a \approx n/10$) both
   methods are $O(n^3)$ and Section 7.2 correctly says "same asymptotic class
   with a $441\times$ smaller constant". The measured growth ($26\times \to
   84\times$) must then be a cache/memory effect, not asymptotics — say so
   consistently.
4. **$k \propto n$ is baked into the synthetic generator.** The growth law
   $k \approx n/21$ is observed at a single real data point ($n=494$) and then
   *imposed* on the synthetic DGP used for the scaling study; the $n=3000$
   conclusions are conditional on it. One sentence acknowledging this, and
   ideally one scaling run with fixed $k$, would close the loop.
5. **Scaling table baseline.** "KKT-Cholesky (no preprocessing)" also needs the
   eigenpairs to assemble $T_0$, so the $3.9\times$ pipeline comparison is
   conservative (it charges preprocessing only to Woodbury). Worth one
   clarifying sentence — it strengthens the result.
6. **CVXPY baseline.** The $5000\times$ figures are dominated by Python-level
   problem construction (1.57 s for an $n=494$ QP). The disclaimer is
   appreciated, but a fairer interior-point reference (direct Clarabel call
   with pre-assembled data, or a `cp.Parameter`-based re-solve) would cost the
   authors little and pre-empt the obvious objection. Reporting one first-order
   QP baseline (e.g. OSQP) would also be informative.
7. **Cache claims.** "Fits in L1/L2" (Remark 4.2, §7.1) is asserted, never
   measured. Either soften to "consistent with", or support with a simple
   experiment (e.g. FLOP-rate comparison across $n_a$).
8. **Timing protocol.** Minimum of three repetitions is acceptable, but please
   also report dispersion, and state the BLAS backend (Accelerate?
   single/multi-threaded?) — at these sub-millisecond scales it matters.
9. **El Karoui edge correction.** Section 3.3 states the correction is
   $O(1/\sqrt{nT})$; the largest-eigenvalue fluctuation scale at the MP edge is
   Tracy–Widom, $O(T^{-2/3})$ for $n/T$ fixed. Please verify the rate and the
   attribution.
10. **Spiked covariance.** Remark 3.2 invokes the spiked model — cite
    Johnstone (2001), and given that signal eigenvalues just above the edge are
    biased upward with imperfect eigenvector overlap (BBP transition), note
    that CRE retains $\lambda_j$ unshrunk for $j \le k$, another point where
    RIE differs.
11. **Notation.** $T$ denotes both the sample size and the target $T_0$;
    $\hat\lambda$ denotes both the MP edge ($\hat\lambda_+$) and the budget
    multiplier in Section 2/Algorithm 1. Both collisions are resolvable with
    trivial renaming.
12. **Proposition 2.1.** "At most $2^n$ realisations" — the argument is fine,
    but the bound is for subsets, of which there are exactly $2^n$; "at most"
    reads oddly. Also state explicitly that the non-degeneracy assumption is
    generic but not verifiable a priori, and that the implementation carries a
    cycling safeguard.
13. **Table 1 (`tab:preprocessing`)** has a column-count mismatch: the tabular
    has five columns (`lrrrr`) but four headers, and the "Speedup" row floats
    oddly. Cosmetic, but it suggests the table was edited late.
14. **Abstract length and density.** The abstract reads like a results section
    (two speedup inventories, a crossover constant, complexity accounting).
    SIFIN abstracts are typically tighter; consider moving the detailed numbers
    to the introduction.
15. **Reproducibility.** The experiments are clearly scripted; please add the
    standard code/data availability statement (and the random seed for the
    synthetic DGP).

---

## Questions for the authors

1. Were the reported timings produced with the dual-feasibility check on
   $2T_0w - \rho\hat\mu$ or on $\tfrac{2}{T}X^\top Xw - \rho\hat\mu$ (M1)? If
   the latter, do the reported portfolios match CVXPY's to optimisation
   accuracy, and do the outer-step counts change once corrected?
2. With the trace-preserving bulk average $\bar\lambda_{\mathrm{bulk}}$ (M3),
   how much do the S&P 500 minimum-variance weights and the reported condition
   number change?
3. Why a 50-point $\rho$-grid rather than tracing the exact frontier with the
   critical line algorithm, given that warm-start efficiency is argued from
   precisely the piecewise-linear structure CLA exploits (M5)?
4. Can the pipeline accommodate a ridge term or factor-aligned linear
   constraints (e.g. sector-neutrality) without losing the Woodbury structure?
   A short remark would widen the applicability claim.

---

*Report prepared for the editor; the technical verifications in M2 (cubic
factorisations) and M6 (norm lower bound) are elementary and can be checked by
hand.*
