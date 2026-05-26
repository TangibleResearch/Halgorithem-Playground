PYTHON ?= python3
VENV ?= .venv
HOST ?= 127.0.0.1
PORT ?= 8000
MANAGE = $(VENV)/bin/python manage.py
PIP = $(VENV)/bin/python -m pip

-include .env
export

.PHONY: help venv install install-dev spacy nltk check test run run-semantic run-stdlib clean

help:
	@echo "halgo2 targets:"
	@echo "  make install       Create .venv, install requirements, spaCy model, and NLTK data"
	@echo "  make install-dev   Install runtime plus pytest"
	@echo "  make run           Run Django site at http://$(HOST):$(PORT)"
	@echo "  make run-semantic  Run Django with model downloads enabled"
	@echo "  make test          Run tests with the fast lexical embedder"
	@echo "  make check         Run Django system checks"
	@echo "  make clean         Remove Python caches"
	@echo ""
	@echo "Optional: put OPENAI_API_KEY=... in .env or paste it into the website textbox."

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt
	$(MAKE) spacy
	$(MAKE) nltk

install-dev: install
	$(PIP) install "pytest>=8"

spacy:
	$(VENV)/bin/python -m spacy download en_core_web_sm

nltk:
	$(VENV)/bin/python -m nltk.downloader wordnet omw-1.4 punkt averaged_perceptron_tagger

check:
	HALGORITHEM_EMBEDDER=lexical $(MANAGE) check

test:
	HALGORITHEM_EMBEDDER=lexical $(VENV)/bin/python -m pytest -q

run:
	HALGORITHEM_EMBEDDER=lexical $(MANAGE) runserver $(HOST):$(PORT)

run-semantic:
	HALGORITHEM_ALLOW_MODEL_DOWNLOAD=1 $(MANAGE) runserver $(HOST):$(PORT)

run-stdlib:
	HALGORITHEM_EMBEDDER=lexical $(VENV)/bin/python server.py --host $(HOST) --port 8765

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
