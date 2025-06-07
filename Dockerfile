# Use NVIDIA NGC PyTorch image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

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

# Default command (can be overridden)
CMD ["torchrun", "--nproc_per_node=1", "train.py"]