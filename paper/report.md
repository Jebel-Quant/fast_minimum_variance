# Referee Report: "Krylov Methods for the Markowitz Portfolio"

**Recommendation:** Major revision

---

## Summary

The paper extends the Krylov-based minimum-variance solver of a companion paper to the full
Markowitz mean-variance problem with general linear equality and inequality constraints. The four
generalisations---general $A$, active-set inequalities, return tilt $\rho\mu^\top w$, and
Ledoit-Wolf ridge $\gamma$---are shown to preserve the symmetric indefinite KKT saddle-point
structure. MINRES is applied matrix-free; a constraint-eliminated CG variant is also developed.
The numerical results on synthetic and S&P 500 data demonstrate large speedups over
CVXPY/Clarabel. The software is clean and the experiments are reproducible. The paper is
well-targeted at a SIAM audience.

---

## Major Comments

**M1. Reliance on unpublished companion paper.** The companion paper [minvar_paper] is cited as
a working paper and is not publicly accessible. Several key results are stated as "shown in
[minvar_paper]" without being reproved here. A referee cannot verify claims that rest on an
inaccessible reference. Either the paper should be made self-contained on the points it logically
depends on, or publication should be coordinated with the companion paper.

**M2. Inconsistent use of $c$ in §6.4.** The notation $c$ was removed from the problem
formulation in §2, but §6.4 still contains the sentence "LW optimises a regularised objective
$c\|Xw\|^2 + \gamma\|w\|^2$." This is inconsistent with the rest of the paper and should be
rephrased in terms of the pre-scaled $X$ that is actually passed to the solver.

**M3. Missing $\|Xw\|$ column in Table 1.** The paper states that the $\|Xw\|$ values in
Table 2 "confirm that all four solvers agree on in-sample portfolio risk." This correctness
check is performed only on the S&P 500 data; no such check is shown for the synthetic benchmark
of Table 1. Either add a $\|Xw\|$ column to Table 1 or move the correctness discussion to §6.4
where Table 2 is introduced.

**M4. Abstract vs. introduction inconsistency.** The abstract states that MINRES "solves
directly" the KKT system, but the introduction (correctly) says that MINRES and CG "approximate
its solution efficiently." These characterisations should be made consistent throughout;
"approximate" is the more honest description.

**M5. Convergence bound (3.2) is a CG bound applied to an indefinite system.** Equation (3.2)
is the standard Chebyshev/CG convergence bound for positive-definite systems. The paper
acknowledges that "for the indefinite saddle-point system the true rate depends on the full
spectral ratio $|\lambda_{\max}(K)|/|\lambda_{\min}(K)|$" and describes (3.2) as a "practical
proxy." However, no justification is given for why $\kappa_\mathrm{eff}$ (the condition number
of the $(1,1)$ block alone) is a reliable proxy for the full spectral ratio of $K$. This needs
either a more careful theoretical argument or an explicit empirical validation (e.g., plotting
observed vs. predicted convergence for one test case).

**M6. Dual feasibility argument (Remark 4.2) is informal and scope-limited.** The argument that
$\nu_i \geq 0$ at termination rests on an appeal to the envelope theorem and the claim that
"loosening a binding constraint can only decrease the objective." While intuitively clear, it is
not a proof: the envelope theorem applies to smooth perturbations of the optimal value function,
and the argument as written does not carefully handle the interaction with the active-set loop or
the case of multiple simultaneously promoted constraints. The scope limitation (acknowledged at
the end of the remark) is important enough to be mentioned in the conclusion.

**M7. Warm-starting discussed four times.** The observation that the saddle-point matrix is
independent of $\rho$ and that warm-starting the active set across frontier points is natural
appears in Remark 4.3, the warm-starting paragraph of §5.2, §6.3, and the conclusion. This
repetition should be consolidated.

---

## Minor Comments

**m1. Self-referential correctness check placement.** The correctness check sentence at the
start of §6.1 refers forward to Table 2. It reads oddly in §6.1 (the synthetic benchmark
section). Either add $\|Xw\|$ to Table 1 (see M3) or move the sentence to §6.4.

**m2. $\kappa_\mathrm{eff} \approx 7.8$ not derived.** Remark 3.1 states this value for $T=2n$
Gaussian data but provides no calculation. A two-line derivation or a footnote with the
Marchenko--Pastur edge values would make this claim verifiable.

**m3. Table 1 footnote: 4 outer steps.** The footnote states "the 4 active-set outer steps"
for the synthetic benchmark with 5 sector caps and long-only constraints. It is not obvious why
there are 4 steps rather than some other number. A brief explanation would help.

**m4. Block Krylov citation.** The suggestion to use block Krylov methods for the efficient
frontier (§6.3) cites [schmelzer2004], a 2004 diploma thesis. A more standard publication
would be more appropriate for a SIAM paper.

**m5. Abbreviation inconsistency.** Table 1 uses "CG (constr.-elim.)" while the text uses
"constraint-eliminated CG" and "CG null-space." One consistent term should be adopted
throughout.

**m6. No guidance on choosing $\rho$.** The return-tilt weight $\rho$ is introduced as a
parameter but the paper gives no guidance on how to set it in practice. Even a one-sentence
remark noting that $\rho$ plays the role of the inverse risk-aversion coefficient and that the
efficient frontier sweeps over a range of $\rho$ would orient practitioners.

**m7. Right panel of Figure 1 uses CG, not MINRES+LW.** The efficient frontier in the right
panel is computed with CG+LW, yet §6.3 and the conclusion emphasise MINRES+LW as the
recommended solver. The panel should use MINRES+LW, or the caption should explicitly state
which solver was used and why.

**m8. Ledoit-Wolf described as "oracle."** The paper calls the shrinkage "oracle linear
shrinkage" in §2 and the introduction. In practice the formula $\gamma = \|X\|_F^2/(n+T)$ is
computed from the sample. The term "oracle" should be clarified or dropped.

**m9. Missing punctuation after displayed Lagrangian.** The displayed equation for
$\mathcal{L}(w,\lambda)$ in §3.1 is not followed by a period or comma before "The stationarity
condition..."

**m10. Table 1 caption footnote ambiguity.** The footnote could more clearly distinguish
between the outer active-set steps (4, reported) and the inner Krylov iterations (not
applicable for the direct solver).

---

## Progress

| Item | Description | Status |
|------|-------------|--------|
| M1 | Reliance on unpublished companion paper | ❌ |
| M2 | Leftover $c$ in §6.4 | ✅ |
| M3 | Missing $\|Xw\|$ column in Table 1 | ✅ |
| M4 | Abstract says "solves directly" vs. introduction "approximate" | ✅ |
| M5 | CG convergence bound applied to indefinite system | ✅ |
| M6 | Dual feasibility argument informal and scope-limited | ✅ |
| M7 | Warm-starting discussed four times | ✅ |
| m1 | Correctness check in §6.1 refers forward to Table 2 | ✅ |
| m2 | $\kappa_\mathrm{eff} \approx 7.8$ not derived | ✅ |
| m3 | Table 1 footnote: 4 outer steps unexplained | ✅ |
| m4 | Block Krylov cites a diploma thesis | ✅ |
| m5 | Abbreviation inconsistency (constr.-elim. vs. null-space) | ✅ |
| m6 | No guidance on choosing $\rho$ | ✅ |
| m7 | Right panel of Figure 1 uses CG, not MINRES+LW | ✅ |
| m8 | "Oracle" shrinkage terminology | ✅ |
| m9 | Missing period after displayed Lagrangian | ✅ |
| m10 | Table 1 footnote conflates outer steps and inner iterations | ✅ |
