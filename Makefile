.PHONY: test coverage clean help build-wheels build-wheel dist upload-test upload release

# Configuration
PYTHON_VERSIONS := 3.11 
DIST_DIR := dist
BUILD_DIR := build

# Default target
help:
	@echo "Available commands:"
	@echo "  Development:"
	@echo "    test          - Run tests with coverage reporting"
	@echo "    coverage      - Update coverage badge in README"
# 	@echo "    build         - Build the Fortran xtension in place"
# 	@echo "    test-only     - Run tests without coverage"
	@echo "    install-dev   - Build and install the package with development dependencies"
# 	@echo "    clean         - Clean coverage and cache files"
# 	@echo ""
# 	@echo "  Building:"
# 	@echo "    build-wheel   - Build wheel for current Python version"
# 	@echo "    build-wheels  - Build cross-platform wheels using cibuildwheel (requires Docker for Linux)"
# 	@echo "    build-macos   - Build macOS wheels only using cibuildwheel"
# 	@echo "    build-linux   - Build manylinux wheels only using cibuildwheel"
# 	@echo "    build-local   - Build wheels locally for available Python versions"
# 	@echo "    dist          - Create source distribution"
# 	@echo ""
# 	@echo "  Release:"
# 	@echo "    upload-test   - Upload to TestPyPI"
# 	@echo "    upload        - Upload to PyPI"
# 	@echo "    release       - Full release (clean, build, test, upload)"


# installing molbar and all the dependencies 
install-dev:
	pip install -r requirements-dev.txt
    git submodule update --init --recursive
	cd submodules/molbar && make install-dev && make install && cd ../../
	rm -rf _skbuild/ ffits.egg-info/ 
	python setup.py build_ext --inplace 
	pip install .
# Run tests with coverage
test:
	pytest 

# Update coverage badge
coverage:
	python -m pytest --cov=ffits --cov-report=xml
	python scripts/update_coverage_badge.py
