FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1
RUN apt-get update 
RUN apt-get install apt-transport-https ca-certificates
RUN apt-get -y install curl
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
RUN apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

RUN git submodule update --init --recursive
RUN pip install torch
RUN if [ "$WITH_TORCHVISION" = "1" ] ; then git clone https://github.com/pytorch/vision.git && cd vision && pip install -v . ; else echo "building without torchvision" ; fi
WORKDIR /opt/pytorch/support
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .


WORKDIR /workspace
RUN chmod -R a+w .

### For custom code change TEST Only ####
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
# Now copy in our code, and run it
COPY . /workspace
RUN python support/setup.py develop
#This should give some result. if it complains CUDA / GPU is not available then its a problem
RUN python test/nms/test_nms.py
RUN python infer_websocket.py -s=voc2007 -b=resnet101 -c='/workspace/model-90000.pth'
