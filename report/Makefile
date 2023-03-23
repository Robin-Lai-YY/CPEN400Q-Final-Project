LATEX_OPTS=-interaction=nonstopmode -shell-escape -pdflua

.PHONY: all clean report presentation

all: report presentation

clean:
	latexmk $(LATEX_OPTS) -C

report:
	latexmk $(LATEX_OPTS) report.tex

presentation:
	latexmk $(LATEX_OPTS) presentation.tex
