FROM python:3.6

WORKDIR /app

# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN pip install pytorch-cpu torchvision-cpu -c pytorch

# Now copy in our code, and run it
COPY . /app

RUN python support/setup.py develop
#RUN python support/setup.py install
RUN python test/nms/test_nms.py

#CMD ["python", "./app/infer_websocket.py", "runserver", "0.0.0.0:8000"]
RUN python infer_websocket.py -s=voc2007 -b=resnet101 -c='/app/model-90000.pth'

#FROM ubuntu:16.04
#ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
#RUN apt-get update && apt-get install -q -y \
#  wget \
#  module-init-tools \
#  build-essential

#RUN cd /opt && \
#  wget $CUDA_RUN && \
#  chmod +x cuda_8.0.44_linux-run && \
#  mkdir nvidia_installers && \
#  ./cuda_8.0.44_linux-run -extract=`pwd`/nvidia_installers && \
#  cd nvidia_installers && \
#  ./NVIDIA-Linux-x86_64-367.48.run -s -N --no-kernel-module

#RUN cd /opt/nvidia_installers && \
#  ./cuda-linux64-rel-8.0.44-21122537.run -noprompt

# Ensure the CUDA libs and binaries are in the correct environment variables
#ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
#ENV PATH=$PATH:/usr/local/cuda-8.0/bin

#RUN cd /opt/nvidia_installers &&\
#    ./cuda-samples-linux-8.0.44-21122537.run -noprompt -cudaprefix=/usr/local/cuda-8.0 &&\
#    cd /usr/local/cuda/samples/1_Utilities/deviceQuery &&\
#    make

#WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery

#FROM continuumio/miniconda3:4.6.14

#RUN conda install -c conda-forge tensorboardx
#RUN conda install -c conda-forge websockets
#RUN conda install -c conda-forge opencv

#RUN conda install --yes \
    #nomkl \
    #Pillow==6.1 \
    #numpy==1.18.0 \
    #opencv-python==3.4.8.29 \
    #protobuf \
    #six \
    #tensorboardX==2.0 \
    #tqdm==4.19.9
    #websockets==8.1

#RUN conda install pytorch-cpu torchvision-cpu -c pytorch
#RUN conda install pytorch-cpu
#WORKDIR /app

# Now copy in our code, and run it
#COPY . /app

#RUN python support/setup.py develop
#RUN python support/setup.py install
#RUN python test/nms/test_nms.py

#CMD ["python", "./app/infer_websocket.py", "runserver", "0.0.0.0:8000"]
#RUN python infer_websocket.py -s=voc2007 -b=resnet101 -c='/app/model-90000.pth'