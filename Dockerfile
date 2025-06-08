# Use NVIDIA NGC PyTorch image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# We can use this one for prod. This image is very big (~22G)
# FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Install C compiler and other build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set environment variables for distributed training
ENV OMP_NUM_THREADS=1
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1

# Copy the data to another directory in case we mount a data volume
RUN cp -r /app/data /app/data_ori

# Default command (can be overridden)
CMD ["torchrun", "--nproc_per_node=1", "train.py"]