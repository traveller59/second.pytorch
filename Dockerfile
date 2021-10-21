# syntax = docker/dockerfile:1.0-experimental

FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ARG UBUNTU_MAJOR_MINOR_VERSION="18.04"
ARG CUDA_MAJOR_MINOR_VERSION="10.2"
ARG CUDNN_MAJOR_VERSION="8"
ARG CUDNN_MAJOR_MINOR_VERSION="${CUDNN_MAJOR_VERSION}.0"
ARG CUDNN_VERSION="${CUDNN_MAJOR_MINOR_VERSION}.5.39"
ARG CUDNN_FULL_VERSION="${CUDNN_VERSION}-1+cuda${CUDA_MAJOR_MINOR_VERSION}"

RUN \
    # update packages
    apt-get update && apt-get upgrade -y \
    # install additional packages
    && DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing --no-install-recommends -y \
    git \
    libboost-all-dev \
    libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_FULL_VERSION} \
    libcudnn${CUDNN_MAJOR_VERSION}-dev=${CUDNN_FULL_VERSION} \
    git \
    wget \
    # make sure these packages don't get upgraded
    && apt-mark hold \ 
    libcudnn${CUDNN_MAJOR_VERSION} \
    libcudnn${CUDNN_MAJOR_VERSION}-dev  \
    # clean up
    && rm -rf /var/lib/apt/lists/*

# install miniconda, adding packages to the base environment
ARG MINICONDA_VERSION="py38_4.10.3"
ARG MINICONDA_PACKAGE="Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
ARG MINICONDA_INSTALL_DIR="/root/miniconda3"
ENV PATH="${MINICONDA_INSTALL_DIR}/bin:${MINICONDA_INSTALL_DIR}/condabin:${PATH}"
WORKDIR /home/docker-user
RUN wget https://repo.anaconda.com/miniconda/${MINICONDA_PACKAGE} \
    && bash ${MINICONDA_PACKAGE} -b -p ${MINICONDA_INSTALL_DIR} \
    && conda update -y -n base -c defaults conda \
    && conda install -c pytorch -c nvidia -c defaults -c conda-forge \
    # packages to be in the base environment
    black \
    cmake \
    cudatoolkit=10.2.* \
    numba \
    numpy \
    opencv \
    openexr-python \
    pip=20.3.* \
    protobuf \
    psutil \
    python=3.8.* \
    pytorch=1.9.* \
    scikit-image \
    seaborn \
    tensorboardx \
    torchvision \
    wheel \
    # configure and clean up
    && conda init \
    && conda config --set report_errors false \
    && conda clean -tipsy \
    && rm ${MINICONDA_PACKAGE}

# install some extra pip packages
RUN pip install \
    fire \
    lupa

# build and install the spconv package
RUN git clone --depth 1 --recursive https://github.com/traveller59/spconv.git \
    && cd spconv \
    && SPCONV_FORCE_BUILD_CUDA=1 python setup.py install \
    && rm -rf spconv
