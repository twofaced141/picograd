#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dir"

build_ptx()
{
    local src="$1" out="$2" entry="$3"

    clang -target nvptx64-nvidia-cuda -march=sm_75 \
          -Xclang -target-feature -Xclang +ptx78 \
          -O3 -ffp-contract=fast -ffreestanding \
          -Wall -Wextra \
          -S "$src" -o "$out"

    sed -i "s/^\.visible \.func ${entry}/.visible .entry ${entry}/" "$out"

    grep -q "^\.visible \.entry ${entry}" "$out" || {
        echo "error: .entry not generated for ${entry}" >&2
        exit 1
    }
}

build_ptx kernels/kernel_sgemm.c sgemm.ptx pg_sgemm_kernel

# second module: all pg_k_* functions become entry points
clang -target nvptx64-nvidia-cuda -march=sm_75 \
      -Xclang -target-feature -Xclang +ptx78 \
      -O3 -ffp-contract=fast -ffreestanding \
      -Wall -Wextra \
      -S kernels/device_kernels.c -o ops.ptx

sed -i 's/^\.visible \.func pg_k_/.visible .entry pg_k_/' ops.ptx
grep -q '^\.visible \.entry pg_k_map' ops.ptx || { echo "ops .entry missing" >&2; exit 1; }

xxd -i sgemm.ptx > sgemm_ptx.h
xxd -i ops.ptx > ops_ptx.h

echo "ok: sgemm.ptx ($(wc -c < sgemm.ptx) bytes), ops.ptx ($(wc -c < ops.ptx) bytes)"