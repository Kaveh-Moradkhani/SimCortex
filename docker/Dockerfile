FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS conda_env

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

COPY docker/simcortex-env.tar.gz /tmp/simcortex-env.tar.gz

RUN mkdir -p /opt/conda-env && \
    tar -xzf /tmp/simcortex-env.tar.gz -C /opt/conda-env && \
    /opt/conda-env/bin/python /opt/conda-env/bin/conda-unpack && \
    chmod -R a+rX /opt/conda-env && \
    rm /tmp/simcortex-env.tar.gz


FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/SimCortex

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    curl \
    tini \
    procps \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libegl1 \
    libopengl0 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

COPY --from=conda_env /opt/conda-env /opt/conda-env

ENV PATH="/opt/conda-env/bin:${PATH}"
ENV PYTHONPATH="/opt/SimCortex/src:/opt/SimCortex"

# Copy only the runtime source files, not the large conda-pack tarball.
COPY --chmod=755 pyproject.toml README.md /opt/SimCortex/
COPY --chmod=755 src /opt/SimCortex/src
COPY --chmod=755 scripts /opt/SimCortex/scripts

# Install SimCortex code without reinstalling heavy dependencies.
RUN python -m pip install --no-cache-dir --no-deps .

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]

CMD ["python", "/opt/SimCortex/scripts/run_pipeline.py", "--help"]
