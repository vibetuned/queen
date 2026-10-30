#! /bin/bash
# Install dependencies for the project
# Install uv, a Python package manager

# Install clang, a C/C++ compiler
#sudo sed -i 's|http://us-east-1.ec2.archive.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
#sudo apt update -y -o Acquire::ForceIPv4=true
sudo apt update -y

sudo apt-get install -y curl unzip

# Navigate to the temporary directory
cd /tmp || exit

# Download the installation file
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Unzip the package (using -q to keep script logs quiet)
unzip -q awscliv2.zip

# Run the install program
sudo ./aws/install

# Clean up the installation artifacts
rm -rf awscliv2.zip aws/

# Return to your previous working directory
cd -
#export HF_HUB_CACHE=/data/hf-cache/hub
#export HF_XET_CACHE=/data/hf-cache/xet


# Mount temporary storage
# Define your paths
S3_URI="s3://.."
LOCAL_DIR="/data/checkpoints"

sudo mkfs -t xfs /dev/nvme1n1
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
sudo chown -R ubuntu:ubuntu /data

# export S3_URI="s3://.."
# export LOCAL_DIR="/data/checkpoints/"
#sudo mkdir -p "$LOCAL_DIR"
#sudo chown -R ubuntu:ubuntu "$LOCAL_DIR"

#ln -s "$LOCAL_DIR" "$HOME/queen/checkpoints"
# aws s3 sync "$S3_URI" "$LOCAL_DIR" --quiet
# aws s3 sync "$LOCAL_DIR" "$S3_URI" --quiet
sudo apt install -y clang

# Install standard Ubuntu utilities for compiling
sudo apt-get install -y build-essential

# Install the Linux headers for your current kernel version
sudo apt-get install -y linux-headers-$(uname -r)

# Install the NVIDIA driver (headless version is best for servers)
sudo apt-get install -y nvidia-headless-595-server-open nvidia-utils-595-server

# Install the CUDA Toolkit
sudo apt-get install -y nvidia-cuda-toolkit

# Update the initramfs to include the new NVIDIA driver
sudo update-initramfs -u

# Set your Git variables
GIT_NAME="Pedro Fillastre"
GIT_EMAIL=""pfillastre@vibetuned.com"

# Apply global configuration
git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

git config --global credential.helper 'cache --timeout=3600'

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Rust toolchain + libhdf5 (pkg-config discovery) for the native prior generator
# (prior-gen/, built by build_prior_v2.sh); tmux for the detached run_*.sh launchers.
sudo apt-get install -y libhdf5-dev pkg-config tmux
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Reboot the instance to load the driver

sudo reboot
