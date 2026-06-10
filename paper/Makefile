PAPER = minvar_paper
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: compile pdf clean experiment

compile: pdf

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(wildcard s*.tex) bib/refs.bib $(wildcard tables/*.tex)
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

experiment:
	cd ../experiment && uv run experiment.py && uv run experiment_synthetic.py && uv run experiment_oos.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.thm *.fls *.fdb_latexmk

