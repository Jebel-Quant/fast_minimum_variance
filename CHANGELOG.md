# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.7.1] - 2026-07-04

### New Features
- Normalize objective by T, make target optional for shrinkage
- Add support for eigenvalue-normalized shrinkage targets and include percentage returns
- Proximal gradient solver, shrinkage utilities, and empirical experiments (#53)
- Add ClusterFuzzLite fuzzing scaffold for fast_minimum_variance (#60)

### Bug Fixes
- *(deps)* Map cvx-linalg package to cvx module in deptry config
- Slice the system operator to the active set once per outer step (#73)

### Documentation
- Add docstring to nested _woodbury helper (#57)

### Dependencies
- *(deps)* Add cvx-linalg dependency

### Maintenance
- Replace identity matrices with `target` for enhanced flexibility in constraints
- Remove redundant notebook, optimize target handling with Cholesky decomposition
- Delete sp500_log_returns.parquet dataset
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Replace np.linalg.cholesky with cvx.linalg.cholesky
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group across 1 directory with 3 updates (#35)
- Sync rhiza template to v0.10.9 (#37)
- Update rhiza to v0.15.1 (#41)
- Update rhiza to v0.15.3 (#43)
- Update rhiza to v0.17.0 (#44)
- Update rhiza to v0.18.4 (#45)
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#47)
- Apply rhiza sync v0.18.4 (#48)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#46)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#49)
- Ignore paper/ folder (local only)
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#52)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#51)
- Update rhiza to v0.18.8 (#50)
- Stop tracking paper .tex files
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#54)
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#56)
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#59)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#58)
- Chore(deps)(deps): bump the github-actions group across 1 directory with 4 updates (#65)
- Close 99.85% -> 100% coverage gap on minvar_problem.py (#68)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group across 1 directory with 3 updates (#66)
- Update rhiza to v1.0.1 (#70)

### Other Changes
- Switch to pct returns, regenerate figures with corrected LW target
- Merge pull request #30 from Jebel-Quant/target
- Merge pull request #34 from Jebel-Quant/dependabot/github_actions/github-actions-8abaa2cbc6
- Merge pull request #33 from Jebel-Quant/dependabot/uv/python-dependencies-4937f16b6a
- Merge pull request #36 from Jebel-Quant/dependabot/github_actions/github-actions-bcb0c4251a
- Update template.yml to remove unused GitHub templates
- Rmt_paper
- Rmt_paper
- Rmt_paper
- Rmt_paper
- Rmt_paper
- Sync Rhiza template v0.18.8 → v0.19.5 (#61)
- Sync Rhiza template v0.19.6 → v0.19.9 (#64)
- Adopt cvx-linalg operators (FactorOperator, SumOperator, operator-aware power_iteration) (#71)
- Balance systems (B, c) and CG speed restoration (#75)
- Remove `data` and `problem` subpackages and unused project artifacts (#76)

## [0.7.0] - 2026-05-07

### Documentation
- Document solve_osqp and fix stale installation instructions

### Dependencies
- *(deps)* Lock file maintenance (#23)

### Maintenance
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Share common marimo benchmark helpers
- Add type annotation to shared run_timed helper
- Annotate mpl parameter in shared notebook helper
- Fix import grouping in figure notebooks

### Other Changes
- Remove 'renovate' from template.yml
- Merge pull request #25 from Jebel-Quant/dependabot/github_actions/github-actions-937d73b4db
- Merge pull request #24 from Jebel-Quant/dependabot/uv/python-dependencies-189a206c4b
- Delete renovate.json
- Remove blank line in _MinVarProblem._solve
- Add solve_osqp using OSQP direct API
- Add solve_osqp to make_minvar_figures benchmarks
- Graphs
- Remove the convex extra
- Remove the convex extra
- Move solve_clarabel and solve_osqp to _BaseProblem
- Add TestSolveClarabel and TestSolveOsqp to test_problem.py
- Add TestSolveCg, TestSolveCvxpy, TestSolveNnls, TestSolveOsqp to test_minvar_problem.py
- Add tests for `solve_clarabel` and `solve_osqp` in `_MinVarProblem` and `_Problem`
- Suppress ty unresolved-attribute errors for clarabel binary extension
- Merge pull request #26 from Jebel-Quant/osqp
- Add PyPI downloads and coverage badges to README
- Add CodeFactor badge to README
- Initial plan
- Merge pull request #28 from Jebel-Quant/copilot/refactor-duplicate-code-in-figures
- Merge pull request #29 from Jebel-Quant/copilot/refactor-duplicate-code-in-figures
- Bump version 0.6.1 → 0.7.0

## [0.6.1] - 2026-05-03

### Other Changes
- Fix docs: point mkdocstrings at fast_minimum_variance instead of missing .api module
- Add all marimo notebooks to mkdocs nav
- Bump version 0.6.0 → 0.6.1

## [0.6.0] - 2026-05-03

### Other Changes
- Refactor `_MinVarProblem`: implement primal-dual loop for long-only constraint and revise documentation
- Add tests for KKT solver and primal-dual implementation; integrate `pd_projected_qp_solver` into the source; add example plots to notebooks
- Remove `pd_projected_qp_solver` implementation and its related tests; refactor test suite to focus on KKT and CVXPY solvers.
- Refactor tests: streamline `test_small.py` and rewrite `test_kkt.py` to focus on KKT vs CVXPY cross-validation with updated fixtures and parametrized cases.
- Refactor `_MinVarProblem`: replace iterative asset elimination with a primal-dual active-set loop, streamline KKT solver, and enhance documentation.
- Add NNLS and CG solvers to `_MinVarProblem` with corresponding tests
- Add Clarabel-based solver to `_MinVarProblem` with tests, expand benchmarks, update documentation and notebooks.
- Update minvar graphs in notebooks (loglog and scaling)
- Fix typecheck errors and add CG convergence analysis to paper
- Remove MINRES solver from notebooks and benchmarks, update display tables, and add new configurations (Clarabel, NNLS).
- Update Markowitz scaling graphs (PDF, PNG) in notebooks.
- Fix invalid TOML key in run_sp500 inline script metadata
- Remove commented-out Clarabel dependency from run_sp500 script
- Fix README mean-variance example: replace np.array([...]) placeholder
- Merge pull request #22 from Jebel-Quant/dual
- Images
- Bump version 0.5.0 → 0.6.0

## [0.5.0] - 2026-05-02

### Other Changes
- Update README.md
- Add MINRES solver support, update literature references, and clean up plotting
- Align type ignore comments in `Problem` class for consistency
- Make clip_and_renormalize a private staticmethod; fix CG crash and type ignore
- Update imports to reflect `problem` module refactor
- Replace `gamma` with `alpha` for ridge regularization; remove redundant `lw_params`
- Add `MinVarProblem` to public API and update visualization assets
- Remove outdated test suite and fixtures for `fast_minimum_variance` module
- Add `MinVarProblem` shrinking solver, refactor figures, and update benchmarks
- Update .gitignore to include `paper/report2.md` and `paper/report3.md`
- Fix LSP violations and adapt call sites to alpha API
- Merge pull request #20 from Jebel-Quant/paper
- Add new dependencies: `requests`, `yfinance`, and additional packages for compatibility
- Add tests for `Problem` factory and rename private classes
- Fix import path for `Problem` in `minvar.py`
- Refactor `paper` scripts for Marimo compatibility
- Migrate `paper` scripts to `book/marimo/notebooks/` for Marimo compatibility
- Expand README: document `MinVarProblem`, new constraints, and Ledoit-Wolf shrinkage.
- Expand README: document `MinVarProblem`, new constraints, and Ledoit-Wolf shrinkage.
- Update paths in notebooks and expand README with `_MinVarProblem` and `_Problem` documentation
- Add `marimo` as a dependency in `run_sp500.py`
- Merge pull request #21 from Jebel-Quant/factory
- Bump version 0.4.0 → 0.5.0

## [0.4.0] - 2026-05-01

### Other Changes
- Add `constraint_active_set` function and tests for active-set inequality handling
- Add `standard` utility function for generating default constraint parameters
- Refactor solvers to use the `API` dataclass, harmonizing interface and simplifying tests.
- Unify portfolio solvers under `API` dataclass implementation
- Move matrix-free operators into API and simplify Krylov solvers
- Fix type annotations and update minvar notebook for API refactor
- Fix README quick-start snippet for API dataclass interface
- Fix stale _util import in doctests for kkt and cvx modules
- Make API dataclass frozen
- Merge pull request #17 from Jebel-Quant/active
- Add tests for kkt_operator and null_space_operator to reach 100% coverage
- Update README to reflect API dataclass interface and general constraints
- Move clip_and_renormalize into api module; make project optional in Krylov solvers
- Add project parameter to solve_kkt and solve_cvxpy; use clip_and_renormalize
- Fix paper scripts for API dataclass refactor; add matplotlib dev dep
- Improve inline documentation for solvers, API handling, and matrix operations
- Add gamma regularization to CVXPY objective; unify LW panel in run_sp500
- Rename API → Problem; add solver methods to Problem class
- Remove standalone solver modules; solvers are Problem methods only
- Remove random.py from library; move make_returns into test conftest
- Make internal Problem methods private
- Add tests to reach 100% coverage
- Update README for new Problem API
- Fix marimo MultipleDefinitionError: move numpy import and R into cell
- Merge pull request #18 from Jebel-Quant/api2
- Reduce api.py by ~40% and fix docs for consolidated module layout
- Bump version 0.3.0 → 0.4.0

## [0.3.0] - 2026-04-30

### Other Changes
- Add paper: Solving the Minimum Variance Portfolio Fast — A Krylov Perspective
- Strengthen active set argument with theoretical note
- Add S&P 500 real-data experiment (495 assets, 5 years)
- Convert to SIAM LaTeX style (siamart.cls + siamplain.bst)
- Exclude `paper/minvar_paper.*` artifacts
- Remove all `minvar_paper.*` artifacts
- Remove all `minvar_paper.*` artifacts
- Add minvar and markowitz tex/pdf, fix gitignore
- Generalise cvx and kkt solvers to full Markowitz problem
- Rename `minvar_cvxpy` to `solve_cvxpy` for clarity and consistency
- Add new tests for return term handling and update test matrix naming
- Rename `minvar_kkt` to `solve_kkt` and update tests for consistency and return term handling
- Rename `minvar_minres` and `minvar_cg` to `solve_minres` and `solve_cg`, generalize solvers to Markowitz problem, and update tests for consistency and return term handling
- Update notebook to use renamed solver functions (`solve_*`) for consistency
- Rename solver functions in `make_figures.py`, `run_sp500.py`, and documentation for consistency (`minvar_*` → `solve_*`)
- Merge pull request #16 from Jebel-Quant/general
- Add Markowitz scaling study, MINRES/CG tests, and references
- Remove unused solver implementations (`solve_minres` and `solve_cg`) and update test to use `solve_minres(R)` directly
- Update `.gitignore` to exclude generated `.pdf` and `.tex` paper files
- Bump version 0.2.1 → 0.3.0

## [0.2.1] - 2026-04-29

### Other Changes
- Update project description in pyproject.toml
- Demote cvxpy to optional extra [convex]
- Add [convex] extra to notebook inline script deps
- Remove invalid known_optional key from deptry config
- Merge pull request #15 from Jebel-Quant/tschm-patch-1
- Bump version 0.2.0 → 0.2.1

## [0.2.0] - 2026-04-29

### Other Changes
- Bump version 0.1.1 → 0.2.0

## [0.1.1] - 2026-04-28

### New Features
- Add marimo notebook and fix trailing newlines
- Add initial draft of "Solving the Minimum Variance Portfolio Fast: A Krylov Perspective" paper
- Add PEP 723 script header to minvar notebook
- Add numpy and cvxpy as runtime dependencies

### Bug Fixes
- Resolve ruff linting errors in minvar notebook
- Normalize minvar_minres weights to sum to 1
- Suppress ty false positive on LinearOperator call-arg
- Add search plugin back to mkdocs.yml plugins list
- Document security exceptions in tests/conftest.py
- Add fast-minimum-variance to notebook script dependencies
- Use text blocks for API signatures in README to pass syntax validation
- Restore minvar_cg after botched merge of implicit Householder functions
- Include 'sum' column in minvar notebook output header

### Documentation
- Add Google-style docstrings with examples to all public functions
- Add minvar notebook to mkdocs nav
- Add API reference page with mkdocstrings
- Write professional README
- Replace block matrix with scalar KKT equations for better GitHub rendering
- Split KKT equations onto separate lines for clearer rendering
- Fix KKT matrix block — add $$ delimiters and correct R vs X
- Use \cr instead of \\ in pmatrix to survive GitHub Markdown parsing
- Remove paper directory
- Remove paper badge and companion paper link from README
- Remove paper entry from mkdocs nav
- Add test and coverage report links to mkdocs nav
- Remove Marimo and Tests documentation from repo and nav

### Performance
- Implicit Householder null-space basis in minvar_cg
- Adjust `make_returns` dimensions to optimize data generation efficiency

### Dependencies
- *(deps)* Update dependency jebel-quant/rhiza to v0.10.4
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.12
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.11.8
- *(deps)* Update dependency astral-sh/uv to v0.11.8

### Maintenance
- Rhiza init — scaffold pyproject.toml, src, and tests
- Sync template files from jebel-quant/rhiza@v0.10.3
- Add cvxpy as dev dependency
- Add mkdocs.yml
- Chore(deps-dev)(deps-dev): Update polars requirement
- Replace CSV loading with synthetic return matrix generator
- Remove pandas dependency and fix deprecated deptry config key
- Remove random forest portfolio approach
- Add scipy as dev dependency
- Remove symmlq solver from minvar notebook
- Update deptry config with package-module name mappings
- Extract Krylov solvers into krylov.py module
- Add tests for random, kkt, krylov, and cvx modules
- Move scipy to runtime deps, trim unused dev deps
- Normalize package name to kebab-case
- Add active-set coverage tests to reach 100% branch coverage
- Improve minvar_minres by replacing explicit KKT matrix with LinearOperator
- Sync rhiza template files
- Retire minvar_minres_lw, extend minvar_minres with c/gamma params
- Extend minvar_cg with c and gamma parameters

### Other Changes
- Initial commit
- Bring in rhiza
- Merge pull request #4 from Jebel-Quant/dependabot/uv/polars-gte-1.40.1
- Merge pull request #5 from Jebel-Quant/solver
- Merge branch 'solver'
- Remove API reference section from README
- Merge remote-tracking branch 'origin/main'
- Merge pull request #7 from Jebel-Quant/minres
- Merge pull request #10 from Jebel-Quant/renovate/jebel-quant-rhiza-0.x
- Merge pull request #11 from Jebel-Quant/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #12 from Jebel-Quant/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #9 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Merge branch 'main' into cg
- Update src/fast_minimum_variance/krylov.py
- Add n_a==1 fast-path in minvar_cg and corresponding test
- Merge pull request #6 from Jebel-Quant/cg
- Merge pull request #14 from Jebel-Quant/wolf
- Bump version 0.1.0 → 0.1.1

<!-- generated by git-cliff -->
