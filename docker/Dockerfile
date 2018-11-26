ARG FROM_IMAGE

FROM ${FROM_IMAGE}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG THIS_USER
ARG THIS_UID

ARG WITH_PROXY
ENV HTTP_PROXY ${WITH_PROXY}
ENV http_proxy ${WITH_PROXY}
ENV HTTPS_PROXY ${WITH_PROXY}
ENV https_proxy ${WITH_PROXY}
ENV CUDA_HOME /usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        wget \
        build-essential \
        curl \
        ffmpeg \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        software-properties-common \
        libasound2-dev \
        libcurl3-dev \
        libhdf5-dev \
        libjack-dev \
        libsndfile-dev \
        libsox-fmt-all \
        python-numpy-dev \
        python-tk \
        pciutils \
        sox \
        swig \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install from pip
RUN pip install -U pip \
                cython
RUN pip install -U numpy \
                pandas \
                h5py \
                scipy \
                matplotlib \
                soundfile \
                transforms3d \
                scikit-learn \
                colorama \
                madmom \
                chainerui \
                python_speech_features

RUN cd /usr/local/lib/python2.7/dist-packages && \
    git clone https://github.com/Fhrozen/BTET.git
RUN pip install cupy
RUN pip install chainer

RUN git clone https://github.com/marsyas/marsyas.git && \
    cd marsyas && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Add user to container
RUN if [ ! -z "${THIS_UID}"  ];then \
    useradd -m -r -u ${THIS_UID} -g root ${THIS_USER}; \
    fi

USER ${THIS_USER}
