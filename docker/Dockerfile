# syntax=docker/dockerfile:1

# --- Backend Stage ---
# Set CUDA version for the backend image.
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=6

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.0-devel-ubuntu22.04 AS backend

# Use https for apt-get
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -o -g ${GROUP_ID} -f user && \
    useradd -o -u $USER_ID -g $GROUP_ID -m -s /bin/bash user

# Setup workspace
RUN mkdir -p /app && chown -R user:user /app
WORKDIR /app

# Setup Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN chown -R user:user /opt/venv /home/user

USER user

# Install PyTorch and common dependencies
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20
ENV PIP_PROGRESS_BAR=off
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu126

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install PyYAML==6.0.2

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install matplotlib==3.9.2

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install numpy==1.26.4

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install tqdm==4.66.4

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install scipy==1.15.2

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install pydicom==3.0.1

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install pandas==2.3.2

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 install fastapi uvicorn python-multipart imageio

RUN --mount=type=cache,target=/home/user/.cache/pip,uid=${USER_ID},gid=${GROUP_ID} \
    pip3 cache purge

# Build ct_laboratory
USER root
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV CUDA_HOME=/usr/local/cuda
RUN git clone https://github.com/tivnanmatt/ct_laboratory.git /ct_laboratory
WORKDIR /ct_laboratory
RUN make clean && \
    make

# Back to app workspace
WORKDIR /app
RUN chown -R user:user /ct_laboratory /app /opt/venv

# Copy backend source
COPY --chown=user:user backend/ /app/backend/

# Set Python path to include reconstruction tools and backend
ENV PYTHONPATH="/app:/ct_laboratory"

USER user
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Frontend Build Stage ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# --- Frontend Serve Stage ---
FROM nginx:alpine AS frontend
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

