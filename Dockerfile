## Base image with Python 3.8
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies (you may need to tweak this based on MimicGen's specific requirements)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \ 
    git \
    libhdf5-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libsm6 \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libxrender-dev \
    libxfixes-dev \
    libxext-dev \
    libx11-dev \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-sync1 \
    libxcb-xkb1 \
    libx11-xcb1 \
    libxi6 \
    libxtst6 \
    libxrender1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    mesa-utils \
    pkg-config \
    sudo \
    x11-apps \
    x11proto-core-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
WORKDIR /cyclic_lxm
COPY . .
RUN pip install --upgrade pip
RUN pip install -e mimicgen
RUN pip install -e robosuite
RUN pip install -e robomimic
RUN pip install -e robosuite-task-zoo
# RUN pip install -e openvla
# RUN pip install -r requirements.txt

