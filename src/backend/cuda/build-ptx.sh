#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dir"

clang -target nvptx64-nvidia-cuda -march=sm_75 \
      -Xclang -target-feature -Xclang +ptx78 \
      -O3 -ffp-contract=fast -ffreestanding \
      -Wall -Wextra \
      -S kernels/kernel_sgemm.c -o sgemm.ptx

sed -i 's/^\.visible \.func pg_sgemm_kernel/.visible .entry pg_sgemm_kernel/' sgemm.ptx

grep -q '^\.visible \.entry pg_sgemm_kernel' sgemm.ptx || {
    echo "error: .entry not generated" >&2
    exit 1
}

xxd -i sgemm.ptx > sgemm_ptx.h
echo "ok: sgemm.ptx ($(wc -c < sgemm.ptx) bytes)"
