#!/usr/bin/env bash
set -ex

# Parse command line arguments
DOWNLOAD_DGN=false
BENCH_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dgn)
      DOWNLOAD_DGN=true
      shift
      ;;
    --bench)
      BENCH_MODE=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Bench mode downloads
if [[ "${BENCH_MODE}" == "true" ]]; then
  REPO_ID="HAMNet/public"

  mkdir -p "/input" "/input/DGN" "/tmp/docker"

  # Download and unzip real_and_GSO.zip under /input
  FILE="real_and_GSO.zip"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "/input"
  unzip -o "/input/${FILE}" -d "/input"
  rm -f "/input/${FILE}"

  # Download and unzip dgn_bench.zip under /input/DGN
  FILE="dgn_bench.zip"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "/input/DGN"
  unzip -o "/input/DGN/${FILE}" -d "/input/DGN/bench"
  rm -f "/input/DGN/${FILE}"

  # Download and unzip dgn_bench.zip under /tmp/docker
  FILE="domain.zip"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "/tmp/docker"
  unzip -o "/tmp/docker/${FILE}" -d "/tmp/docker"
  rm -f "/tmp/docker/${FILE}"

  # Download and unzip bench_eps.zip under /tmp/docker
  FILE="bench_eps.zip"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "/tmp/docker"
  unzip -o "/tmp/docker/${FILE}" -d "/tmp/docker"
  rm -f "/tmp/docker/${FILE}"

  exit 0
fi

# Download DGN (optional)
if [[ "${DOWNLOAD_DGN}" == "true" ]]; then
  echo "Downloading DGN dataset..."

  hf download "imm-unicorn/corn-public" "DGN.tar.gz" \
  --repo-type model \
  --local-dir "/input/DGN"
    
  cd /input/DGN
  tar -xzf "DGN.tar.gz"
  rm -f "DGN.tar.gz"
  cd -
else
  DEST_DIR="/tmp/docker"
  REPO_ID="HAMNet/public" # dataset repo id

  mkdir -p "${DEST_DIR}"

  # Download episode for training
  FILE="eps-fr3-near-and-rand-1024x384.pth"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  # Download episode for testing
  FILE="eps-demo-16x8.pth"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  # Download sysid
  FILE="new_sysid.pkl"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"

  DEST_DIR="/input/robot"
  mkdir -p "${DEST_DIR}"

  # Download robot cloud (Optional)
  FILE="custom_v3_cloud.pkl"
  hf download "${REPO_ID}" \
    --repo-type dataset \
    --include "${FILE}" \
    --local-dir "${DEST_DIR}"
fi
