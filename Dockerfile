# Base image with Python 3.8
FROM --platform=linux/amd64 python:3.8-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a non-root user
RUN useradd -ms /bin/bash op

# Install system dependencies (you may need to tweak this based on MimicGen's specific requirements)
RUN apt-get update && apt-get install -y \
    gcc \ 
    git \
    libhdf5-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    pkg-config \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add the 'op' user to the sudoers file to give it sudo privileges
RUN echo "op ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER op
WORKDIR /home/op

RUN pip install --upgrade pip

# # Clone MimicGen repository
# RUN git clone https://github.com/NVlabs/mimicgen.git /app/mimicgen

# # Set up Python environment and install dependencies
# RUN pip install --upgrade pip \
#     && pip install -e /app/mimicgen

# # Install additional dependencies (MuJoCo, robosuite, etc.)
# RUN pip install mujoco==2.3.2
# RUN git clone https://github.com/ARISE-Initiative/robosuite.git /app/robosuite \
#     && pip install -e /app/robosuite

# RUN git clone https://github.com/ARISE-Initiative/robomimic.git /app/robomimic \
#     && pip install -e /app/robomimic

# # Set entrypoint to run MimicGen scripts
# CMD ["python", "/app/mimicgen/scripts/your_script.py"]