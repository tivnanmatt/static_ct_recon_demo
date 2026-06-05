#!/bin/bash
set -e

# Use the virtual environment
export PATH="/opt/venv/bin:$PATH"

echo "Installing system dependencies for GDCM..."
apt-get update
apt-get install -y libgdcm-tools

echo "Installing Python DICOM decoding plugins..."
# pydicom 3.x needs these for JPEG/GDCM support
# We use versions compatible with numpy 1.26.4
pip install "pylibjpeg>=1.4.0" "pylibjpeg-libjpeg>=1.3.0" "pylibjpeg-openjpeg>=1.3.0"

echo "DICOM decoding environment setup complete."
