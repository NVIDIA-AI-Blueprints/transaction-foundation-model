#!/usr/bin/env bash
set -euo pipefail

DATA_DISK="/dev/disk/by-id/google-${data_disk_name}"
MOUNT_DIR="/mnt/tfm"
MARKER_DIR="/opt/tfm"
MARKER_FILE="$MARKER_DIR/bootstrap.done"
REBOOT_MARKER="$MARKER_DIR/rebooted-after-driver-install"

mkdir -p "$MARKER_DIR"
rm -f "$MARKER_FILE"
echo "${notebook_image}" > "$MARKER_DIR/notebook-image"
echo "${bucket_name}" > "$MARKER_DIR/artifacts-bucket"

if [[ "${auto_shutdown_minutes}" != "0" ]]; then
  shutdown -h +"${auto_shutdown_minutes}" "Auto-stopping GPU notebook VM after ${auto_shutdown_minutes} minutes to cap cost." || true
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  gnupg \
  jq \
  rsync \
  git \
  git-lfs \
  python3 \
  docker.io

systemctl enable --now docker

if [[ -b "$DATA_DISK" ]]; then
  if ! blkid "$DATA_DISK" >/dev/null 2>&1; then
    mkfs.ext4 -F "$DATA_DISK"
  fi

  mkdir -p "$MOUNT_DIR"
  if ! mountpoint -q "$MOUNT_DIR"; then
    mount "$DATA_DISK" "$MOUNT_DIR"
  fi

  if ! grep -q "$DATA_DISK" /etc/fstab; then
    echo "$DATA_DISK $MOUNT_DIR ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
fi

mkdir -p "$MOUNT_DIR"/{workspace,data,models,artifacts,tmp}
chmod -R 0777 "$MOUNT_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p /opt/google/cuda-installer
  cd /opt/google/cuda-installer
  curl -fSsL -o cuda_installer.pyz \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
  python3 cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod || true
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

if ! dpkg -s google-cloud-ops-agent >/dev/null 2>&1; then
  cd /tmp
  curl -fsSLO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
fi
systemctl enable --now google-cloud-ops-agent

git lfs install --system || true
touch "$MARKER_FILE"

if ! nvidia-smi >/dev/null 2>&1 && [[ ! -f "$REBOOT_MARKER" ]]; then
  touch "$REBOOT_MARKER"
  reboot
fi
