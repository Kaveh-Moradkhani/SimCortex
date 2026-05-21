FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/simcortexpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    curl \
    tini \
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

COPY docker/simcortexpp-env.tar.gz /tmp/simcortexpp-env.tar.gz

RUN mkdir -p /opt/conda-env && \
    tar -xzf /tmp/simcortexpp-env.tar.gz -C /opt/conda-env && \
    /opt/conda-env/bin/python /opt/conda-env/bin/conda-unpack && \
    rm /tmp/simcortexpp-env.tar.gz

ENV PATH="/opt/conda-env/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY . /opt/simcortexpp

RUN python -m pip install --no-cache-dir .

ENV PATH="/opt/conda-env/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["scpp", "--help"]
