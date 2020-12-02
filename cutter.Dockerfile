FROM ubuntu:18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update && apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    iputils-ping \
    python3-dev \
    python3-pip

RUN apt-get -y update && apt-get install -y \
    gfortran \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    libgl1-mesa-glx

RUN apt-get -y update && apt-get install -y \
    python3-bs4

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install --no-cache-dir opencv-python
