# Referee Report: "Krylov Methods for the Markowitz Portfolio"

**Recommendation:** Major Revision

---

## Summary

The paper proposes applying MINRES and conjugate gradient (CG) methods to the KKT
saddle-point system arising in mean-variance portfolio optimisation, combined with an
active-set constraint-handling strategy and a Ledoit-Wolf-style regularisation heuristic.
The central claim is that this yields order-of-magnitude speedups over CVXPY on benchmark
problems (up to $1326\times$ on a synthetic $n=1000$ problem, $71\times$ on S\&P 500
data).  The topic is legitimate and the computational results appear promising, but the
manuscript has significant theoretical gaps, several claims that are either imprecise or
unsupported, and a number of internal inconsistencies that must be resolved before
publication.

---

## Major Comments

**M1. Misattribution of convergence bound (§3).**
The proxy bound (3.3) for MINRES on the indefinite system $K$ applies a factor
$\lfloor k/2 \rfloor$ relative to the SPD Chebyshev bound (3.2), attributed to
Paige-Saunders \[paige1975\].  That 1975 paper derives the algorithm and establishes
basic error bounds but does not contain this specific $\lfloor k/2 \rfloor$
polynomial-degree argument for symmetric indefinite systems.  The correct source for
MINRES convergence on indefinite systems is more recent (e.g., Choi 2006, or Greenbaum
1997).  The authors must either cite the correct source or provide a self-contained
derivation.  Using an uncited bound as a central theoretical tool is not acceptable.

**M2. The "Ledoit-Wolf" regularisation is not the Ledoit-Wolf estimator (§2).**
The paper uses $\gamma = \|X\|_F^2/(n+T)$ with the rescaling $\sqrt{T/(n+T)}\,X$ passed
to the solver, citing \[ledoit2004\].  The Ledoit-Wolf (2004) shrinkage estimator is an
analytically optimal convex combination of the sample covariance and a structured target,
with a shrinkage intensity estimated from data.  The formula presented here is a fixed-weight
regulariser whose derivation is not given and whose connection to Ledoit-Wolf 2004 is not
established.  Either (a) derive this formula from first principles and cite it correctly, or
(b) demonstrate rigorously that it coincides with or approximates the Ledoit-Wolf estimator
for the specific problem structure at hand.  As written, the citation is misleading and the
heuristic is unsubstantiated.

**M3. Active-set method: missing treatment of degenerate constraints (§4).**
The paper proves dual feasibility via a parametric value function argument but explicitly
acknowledges a scope limitation: constraints that become binding without being violated
require a multiplier sign check that the current algorithm does not perform.  This is
precisely the case that can cause the active-set method to stall in practice.  The paper
must either (a) extend the algorithm to handle this case and prove correctness, or (b)
provide a rigorous characterisation of problem instances on which the limitation does not
arise, and demonstrate that all reported benchmarks satisfy this condition.

**M4. Inconsistency between §6.3 and Figure 1 caption.**
Section 6.3 states that the efficient frontier timing benchmark uses 21 equally spaced
values of $\rho \in [0, 2]$, yet the Figure 1 caption explicitly states "tracing 51
risk-return points."  If the figure is illustrative and uses 51 points while the timing
uses 21, this must be stated unambiguously in both places.  As written, a reader cannot
determine which number the reported times (MINRES+LW $0.043$ s, $838\times$ speedup)
correspond to, which undermines reproducibility.

**M5. Relative error claim is inconsistent with the reported numbers (§6.4, Table 2).**
The text claims MINRES agrees with CVXPY to "0.7% relative error" in portfolio cost on
the S\&P 500 experiment.  The reported $\|Xw\|$ values are $0.2255$ (MINRES) versus
$0.2246$ (CVXPY/KKT/CG), a discrepancy of approximately $0.4\%$ in $\|Xw\|$.  The $0.7\%$
figure appears to refer to the variance $\|Xw\|^2$, but this is not stated.  The authors
must (a) specify which quantity the relative error refers to, (b) use consistent units
throughout, and (c) discuss whether this accuracy level is acceptable for the claimed
application.

**M6. CG null-space cost analysis is incomplete (§5.4).**
The paper notes that forming the null-space basis via QR of the active constraint matrix
costs $O(n^2 m_\mathcal{A})$, reaching $O(n^3)$ for dense $A$.  No analysis is given of
when this cost dominates the Krylov solve, nor is the CG+LW benchmark timing broken down
to show the QR vs. iterative-solve split.  The S\&P 500 Table 2 shows CG+LW at $0.069$ s
versus MINRES+LW at $0.021$ s — a $3\times$ disadvantage — but this gap is not explained.
A cost breakdown is required.

---

## Minor Comments

**m1. Direct solver benchmark (§5.1 and Table 1).** The paper notes that the
Bunch-Kaufman direct solver is slower than `numpy.linalg.solve` (LU) at moderate sizes.
For a symmetric indefinite system, LU without pivoting designed for general systems is
theoretically less appropriate than Bunch-Kaufman; the benchmark comparison should
document which LAPACK routine each path calls and whether the advantage of LU here is an
artifact of overhead rather than arithmetic cost.

**m2. Missing cross-reference.** The figure caption references `\S\ref{sec:frontier}`
but the label `\label{sec:frontier}` does not appear to be defined in the manuscript.
This will produce a broken reference in the compiled PDF and must be corrected.

**m3. Companion paper inaccessible.** Reference \[minvar\_paper\] is an unpublished
working paper.  SIAM policy generally requires that citations be accessible to referees
and readers.  The authors should provide a preprint (e.g., on arXiv) or restructure
claims that rely on \[minvar\_paper\] to be self-contained.

**m4. Marchenko-Pastur approximation (§3, Remark 3.1).** The derivation of
$\kappa_\mathrm{eff} \approx 7.8$ for $T = 2n$ uses the Marchenko-Pastur law, which
holds asymptotically under i.i.d. Gaussian assumptions.  The paper does not state these
assumptions explicitly and does not discuss their applicability to financial return data
with heavy tails and cross-sectional correlations.  The Gaussian assumption should be
stated clearly, and the discrepancy with S\&P 500 iteration counts (424 vs. 32) should be
attributed to this gap.

**m5. Speedup figures depend on CVXPY version and hardware.** The abstract's $1326\times$
and $838\times$ speedups depend on CVXPY's canonicalisation overhead, which may not hold
across CVXPY versions or machines.  The claim that "CVXPY constructs the quadratic cost
matrix $P = 2X^\top X$ during problem canonicalisation" may not hold for all Clarabel
backends.  CVXPY and Clarabel version numbers should be reported, and speedup figures
should be accompanied by a note on their hardware and software dependence.

**m6. Notation: $\rho$ as "inverse risk-aversion."** Section 2 states "$\rho$ is the
inverse risk-aversion coefficient."  In standard mean-variance theory, the risk-aversion
parameter $\lambda$ appears in $\min \sigma^2 - \lambda \mu^\top w$, so $\rho = \lambda$
is risk-aversion (not its inverse).  "Inverse risk-aversion" typically means $1/\lambda$.
The convention here is the standard one; the label is wrong and should be corrected.

**m7. §6.3 frontier verification is vague.** The claim "the Krylov solvers produce an
identical curve, verified by comparing portfolio weights to relative tolerance $10^{-4}$"
does not state which norm, which solver, or over how many frontier points.  A one-line
table or explicit statement would suffice.

**m8. No missing-period or out-of-sample analysis.** All results are in-sample ($T$
returns, same data used for $X$ and optimisation).  A brief note acknowledging the
absence of out-of-sample evaluation would prevent misinterpretation by practitioners.

---

## Progress

| Item | Description | Status |
|------|-------------|--------|
| M1 | $\lfloor k/2 \rfloor$ factor misattributed to \[paige1975\] | ✅ |
| M2 | LW formula not matched to \[ledoit2004\] | ✅ |
| M3 | Active-set scope limitation: degenerate constraints not handled | ✅ |
| M4 | 21-point vs. 51-point inconsistency in §6.3 / Figure 1 caption | ✅ |
| M5 | 0.7% relative error claim inconsistent with reported numbers | ✅ |
| M6 | CG null-space cost breakdown missing | ❌ |
| m1 | Bunch-Kaufman vs. LU benchmark discussion | ❌ |
| m2 | Missing `\label{sec:frontier}` | ✅ |
| m3 | Companion paper \[minvar\_paper\] inaccessible | ❌ |
| m4 | Marchenko-Pastur Gaussian assumption not stated explicitly | ✅ |
| m5 | CVXPY version/hardware dependence of speedup not documented | ✅ |
| m6 | $\rho$ labelled "inverse risk-aversion" but is risk-aversion | ✅ |
| m7 | Frontier verification claim lacks specifics (norm, solver, count) | ❌ |
| m8 | No note on absence of out-of-sample evaluation | ❌ |
