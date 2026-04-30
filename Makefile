## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DEFAULT_AI_MODEL=claude-sonnet-4.6
LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: fix quoting bug and typo (mkdocstring -> mkdocstrings)
MKDOCS_EXTRA_PACKAGES = --with-editable . --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk

.PHONY: paper
paper: ## Build the LaTeX paper to PDF (pdflatex + bibtex + pdflatex x2)
	cd paper && \
	  pdflatex -interaction=nonstopmode minvar_paper.tex > /dev/null && \
	  bibtex minvar_paper && \
	  pdflatex -interaction=nonstopmode minvar_paper.tex > /dev/null && \
	  pdflatex -interaction=nonstopmode minvar_paper.tex | grep -E "Output written|Warning|Error" || true
