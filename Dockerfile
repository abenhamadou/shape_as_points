# Start from a PyTorch + CUDA compatible base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app


# Remove problematic NVIDIA repos to avoid GPG errors
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list

# Fix NVIDIA repo GPG key issue - do this BEFORE any apt-get operations
#RUN rm -f /etc/apt/sources.list.d/cuda.list && \
#    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8-dev \
    build-essential \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    wget \
    git \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    bzip2 \
    curl \
    pkg-config \
    unzip \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# --- Set CUDA environment variables ---
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# --- Install micromamba (minimal, fast, conda-compatible) ---
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ bin/micromamba --strip-components=1

# --- Set environment variables for micromamba ---
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
SHELL ["/bin/bash", "-c"]


# --- Copy environment.yml ---
COPY environment.yaml /app/environment.yaml

# --- Create environment from environment.yml ---
RUN micromamba create -y -f /app/environment.yaml && micromamba clean -a -y

# --- Verify CUDA is available ---
RUN nvcc --version && echo "CUDA compilation tools available"


# Copy requirements and install pip packages
COPY requirements.txt /app/requirements.txt
RUN micromamba run -n sap pip install --no-cache-dir -r /app/requirements.txt \
    --exists-action=w \
    --disable-pip-version-check \
    --prefer-binary \
    --upgrade-strategy only-if-needed

# Install pytorch-scatter
RUN micromamba install -n sap -y -c pyg pytorch-scatter

# --- Verify PyTorch3D GPU support ---
RUN micromamba run -n sap python -c "\
import torch; \
print('PyTorch CUDA available:', torch.cuda.is_available()); \
print('CUDA version:', torch.version.cuda); \
import pytorch3d; \
print('PyTorch3D version:', pytorch3d.__version__); \
from pytorch3d import _C; \
print('PyTorch3D CUDA support:', hasattr(_C, 'knn_points_idx'))"

# Copy application code
COPY . /app


# git clone ...
#cd /app/pointnet2/pointnet2_ops_lib


# --- Default environment on container start ---
ENTRYPOINT ["micromamba", "run", "-n", "sap"]
CMD ["python"]

