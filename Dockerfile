FROM python:3.11-slim

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /build

# Build
RUN pip install --upgrade pip build scikit-build numpy && \
    python setup.py build_ext --inplace && \
    python -m build --wheel

# Output wheel to /dist
RUN mkdir -p /dist && cp dist/*.whl /dist/

VOLUME ["/dist"]
CMD ["python", "-m", "build", "--wheel"]
