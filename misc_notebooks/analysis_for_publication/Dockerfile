ARG BASE_IMAGE="yfukai/conda-jax:latest-cuda"

FROM ${BASE_IMAGE}
#https://stackoverflow.com/questions/44438637/arg-substitution-in-run-command-not-working-for-dockerfile
ARG BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Installing necessary packages

# Installing basicpy and other pip packages
RUN pip --no-cache-dir install jupyterlab git+https://github.com/yfukai/BaSiCPy@autosegment

RUN apt-get install -y openjdk-8-jdk-headless -qq > /dev/null && \
    apt-get install maven &&\
    cd /opt &&\
    wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip && unzip fiji-linux64.zip &&\
    cd Fiji.app/ &&\
    ./ImageJ-linux64 --update add-update-site BaSiC https://sites.imagej.net/BaSiC/ &&\
    ./ImageJ-linux64 --update update

ENTRYPOINT [ "bash" ]