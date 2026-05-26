# Shared ML_sdfi image — all four sibling repos required in build context.
# Build from parent directory containing ML_Production, ML_geo_production,
# multi_channel_dataset_creation, and ML_sdfi_fastai2:
#   docker build -f ML_Production/Dockerfile -t ml_sdfi:latest .
#
# Pre-check (recommended):
#   /path/to/orchestrator/check_docker_build_context.sh .

FROM condaforge/mambaforge:24.11.3-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_ENV_NAME=ML_sdfi

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy all four sibling repos (build context = parent directory)
COPY ML_Production /workspace/repos/ML_Production
COPY ML_geo_production /workspace/repos/ML_geo_production
COPY multi_channel_dataset_creation /workspace/repos/multi_channel_dataset_creation
COPY ML_sdfi_fastai2 /workspace/repos/ML_sdfi_fastai2

# Create conda env and install (same steps as README, from ML_Production root)
RUN cd /workspace/repos/ML_Production && \
    mamba env create -f environment.yml && \
    bash -lc 'source /opt/conda/etc/profile.d/conda.sh && \
      conda activate ML_sdfi && \
      export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib && \
      export INSTALL_PYTORCH_NO_GPU=1 && \
      bash install_pytorch.sh && \
      pip install --pre --no-build-isolation -r requirements_pip.txt && \
      bash install_local_repos.sh && \
      pip install -r requirements_extra.txt && \
      gdal_ver=$(gdal-config --version) && \
      pip install --force-reinstall --no-cache-dir gdal==${gdal_ver}.* && \
      mamba install -y -c conda-forge --force-reinstall libtiff libjpeg-turbo libdeflate'

# Shared models symlink (same as previous ML_Production Docker layout)
RUN rm -rf /workspace/repos/ML_geo_production/models && \
    ln -s ../ML_Production/models /workspace/repos/ML_geo_production/models

ENV PATH="/opt/conda/envs/ML_sdfi/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/conda/envs/ML_sdfi/lib" \
    CONDA_DEFAULT_ENV=ML_sdfi \
    GTIFF_SRS_SOURCE=EPSG

WORKDIR /workspace/repos/ML_Production

CMD ["/bin/bash"]
