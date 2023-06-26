ARG BASE_IMAGE="yfukai/conda-jax:latest-cuda"

FROM ${BASE_IMAGE}
#https://stackoverflow.com/questions/44438637/arg-substitution-in-run-command-not-working-for-dockerfile
ARG BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Installing necessary packages

# Installing basicpy and other pip packages
RUN pip --no-cache-dir install jupyterlab git+https://github.com/yfukai/BaSiCPy@autosegment
