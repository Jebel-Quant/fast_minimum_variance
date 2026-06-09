PAPER = minvar_paper
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all pdf clean experiment

all: pdf

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(wildcard s*.tex) bib/refs.bib $(wildcard tables/*.tex)
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

experiment:
	cd ../experiment && uv run experiment.py && uv run experiment_synthetic.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.thm *.fls *.fdb_latexmk

