# Makefile for "PANORAMA paper"
#

# ----- Sources -----

NOTEBOOK = paper.ipynb

BIBLIOGRAPHY = references.bib

TEMPLATE = bioRxiv
# ----- Tools -----
.ONESHELL:

# Root directory
ROOT = $(shell pwd)

# ----- Top-level targets -----
# Base commands
PYTHON = python                            # specific version for talking to compute cluster
JUPYTER = jupyter
JUPYTER_NBCONVERT = $(JUPYTER) nbconvert
PDFLATEX = pdflatex
BIBTEX = bibtex
MAKEINDEX = makeindex
PIP = pip

RSYNC = rsync
TR = tr
CAT = cat
SED = sed
RM = rm -rf
CP = cp
MV = mv
CHDIR = cd
MKDIR = mkdir -p
ZIP = zip -r
UNZIP = unzip
WGET = wget
ECHO = echo

# Construction variable
BUILD_DIR = $(ROOT)/build
# LaTex construction
LATEX_PAPER_STEM = paper
LATEX_PAPER = $(LATEX_PAPER_STEM).tex

# Default prints a help message
help:
	@make usage

.PHONY: build
build:
	$(MKDIR) $(BUILD_DIR)

build_latex: build
	$(JUPYTER_NBCONVERT) --to latex $(NOTEBOOK) --template=bioRxiv --output-dir=$(BUILD_DIR) --output=$(LATEX_PAPER_STEM) --no-input

.PHONY: latex
latex: clean build_latex
	$(MV) $(BUILD_DIR)/$(LATEX_PAPER) $(ROOT)/

pdflatex:
	$(PDFLATEX) -output-directory=$(BUILD_DIR) $(BUILD_DIR)/$(LATEX_PAPER_STEM)

bibtex:
	$(CP) $(BIBLIOGRAPHY) $(BUILD_DIR)
	$(CHDIR) $(BUILD_DIR)
	$(BIBTEX) $(LATEX_PAPER_STEM)
	$(CHDIR) $(ROOT)

.PHONY: pdf
pdf: clean build build_latex pdflatex bibtex pdflatex pdflatex
	$(MV) $(BUILD_DIR)/$(LATEX_PAPER_STEM).pdf $(ROOT)/

# Clean up the build
clean:
	$(RM) $(BUILD_DIR)

## Clean up everything, including the venv (which is quite expensive to rebuild)
#reallyclean: clean
#	$(RM) $(VENV)

# ----- Usage -----

define HELP_MESSAGE
Initialisation:
	make init		get all the required packages and create a panorame_paper conda environment

Editing:
   make live        run the notebook server

Production:
   make latex       build the paper using Jupyter nbconvert
   make pdf        	build a PDF of the paper for bioRxiv

Maintenance:
   make clean       clean-up the build

endef
export HELP_MESSAGE

usage:
	@echo "$$HELP_MESSAGE"