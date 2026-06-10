PAPER = rmt_paper
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: compile pdf clean experiment

compile: pdf clean

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(wildcard s*.tex) ../paper/bib/refs.bib $(wildcard tables/*.tex)
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

experiment:
	cd ../experiment && uv run experiment_rmt.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.thm *.fls *.fdb_latexmk

