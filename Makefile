# Makefile

define HELP_MESSAGE
ksim-kbot

# Installing

1. Create a new Conda environment: `conda create --name ksim-kbot python=3.11`
2. Activate the environment: `conda activate ksim-kbot`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#        PyPI Build        #
# ------------------------ #

build-for-pypi:
	@pip install --verbose build wheel twine
	@python -m build --sdist --wheel --outdir dist/ .
	@twine upload dist/*
.PHONY: build-for-pypi

push-to-pypi: build-for-pypi
	@twine upload dist/*
.PHONY: push-to-pypi

# ------------------------ #
#       Static Checks      #
# ------------------------ #

format:
	@black ksim_kbot --exclude "run_*"
	@ruff format ksim_kbot --exclude "run_*"
	@ruff check --fix ksim_kbot --exclude "run_*"
.PHONY: format

static-checks:
	@black --diff --check ksim_kbot --exclude "run_*"
	@ruff check ksim_kbot --exclude "run_*"
	@mypy --install-types --non-interactive ksim_kbot --exclude "run_*"
.PHONY: static-checks

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
