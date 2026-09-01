.PHONY: help install-dev test test-unit test-integration coverage regenerate-examples clean

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIST_DIR := dist

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
help:
	@echo "Available commands:"
	@echo "  install-dev      - Build and install the package with development dependencies"
	@echo "  test             - Run the full test suite"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  coverage         - Run tests and generate a coverage report (coverage.xml)"
	@echo "  regenerate-examples - Regenerate tests/examples/ fixtures from their raw structures (needs xtb+molbar)"
	@echo "  clean            - Remove build artifacts and caches"

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

# Install all dependencies, then build and install ffits in-place
install-dev:
	pip install -r requirements-dev.txt
	rm -rf _skbuild/ ffits.egg-info/
	python setup.py build_ext --inplace
	pip install .

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run the full test suite
test:
	pytest tests/

# Run unit tests only
test-unit:
	pytest tests/unit_tests

# Run integration tests only
test-integration:
	pytest tests/integration_tests

# Run tests with coverage report and update the README badge
coverage:
	pytest tests/ --cov=ffits --cov-report=xml --cov-report=term
	python scripts/update_coverage_badge.py

# Regenerate the derived example fixtures under tests/examples/ (see
# tests/examples/README.md for what's raw vs. derived). Requires ffits,
# xtb, and molbar to be installed and on PATH/importable.
regenerate-examples:
	python scripts/regenerate_examples.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf _skbuild/ ffits.egg-info/ build/ $(DIST_DIR)/
	rm -rf .pytest_cache/ .coverage coverage.xml
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +
