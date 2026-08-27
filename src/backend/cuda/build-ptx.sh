#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"
cd "$dir"

build_ptx()
{
    local src="$1" out="$2" entry="$3"
    local out_tmp
    out_tmp=$(mktemp)
    # try ptx78 first (ignore warning, still generates valid PTX on older clang)
    if clang -target nvptx64-nvidia-cuda -march=sm_75 \
             -Xclang -target-feature -Xclang +ptx78 \
             -O3 -ffp-contract=fast -ffreestanding \
             -Wall -Wextra \
             -S "$src" -o "$out" 2> "$out_tmp.log"; then
        # success even if warning about unrecognized feature, filter it
        grep -v "not a recognized feature" "$out_tmp.log" >&2 || true
        rm -f "$out_tmp" "$out_tmp.log"
    else
        # fallback: try lower ptx versions
        cat "$out_tmp.log" >&2
        local ptx=""
        for ver in 75 74 73 72 71 70 65 64 63; do
            if clang -target nvptx64-nvidia-cuda -march=sm_75 \
                     -Xclang -target-feature -Xclang +ptx${ver} \
                     -O3 -ffp-contract=fast -ffreestanding \
                     -Wall -Wextra \
                     -S "$src" -o "$out_tmp" 2> "$out_tmp.log"; then
                ptx="$ver"
                break
            fi
        done
        if [ -z "$ptx" ]; then
            echo "error: no suitable ptx version found" >&2
            cat "$out_tmp.log" >&2
            rm -f "$out_tmp" "$out_tmp.log"
            exit 1
        fi
        clang -target nvptx64-nvidia-cuda -march=sm_75 \
              -Xclang -target-feature -Xclang +ptx${ptx} \
              -O3 -ffp-contract=fast -ffreestanding \
              -Wall -Wextra \
              -S "$src" -o "$out" 2> "$out_tmp.log" || {
            cat "$out_tmp.log" >&2
            rm -f "$out_tmp" "$out_tmp.log"
            exit 1
        }
        echo "warn: using ptx$ptx (ptx78 not supported by this clang)" >&2
        rm -f "$out_tmp" "$out_tmp.log"
    fi

    sed -i "s/^\.visible \.func ${entry}/.visible .entry ${entry}/" "$out"

    grep -q "^\.visible \.entry ${entry}" "$out" || {
        echo "error: .entry not generated for ${entry}" >&2
        exit 1
    }
}

build_ptx kernels/kernel_sgemm.c sgemm.ptx pg_sgemm_kernel

# second module: all pg_k_* functions become entry points
{
    out_tmp=$(mktemp)
    if clang -target nvptx64-nvidia-cuda -march=sm_75 \
             -Xclang -target-feature -Xclang +ptx78 \
             -O3 -ffp-contract=fast -ffreestanding \
             -Wall -Wextra \
             -S kernels/device_kernels.c -o ops.ptx 2> "$out_tmp.log"; then
        grep -v "not a recognized feature" "$out_tmp.log" >&2 || true
        rm -f "$out_tmp" "$out_tmp.log"
    else
        cat "$out_tmp.log" >&2
        ptx=""
        for ver in 75 74 73 72 71 70 65 64 63; do
            if clang -target nvptx64-nvidia-cuda -march=sm_75 \
                     -Xclang -target-feature -Xclang +ptx${ver} \
                     -O3 -ffp-contract=fast -ffreestanding \
                     -Wall -Wextra \
                     -S kernels/device_kernels.c -o "$out_tmp" 2> "$out_tmp.log"; then
                ptx="$ver"
                break
            fi
        done
        if [ -z "$ptx" ]; then echo "error: no ptx for ops" >&2; cat "$out_tmp.log" >&2; rm -f "$out_tmp" "$out_tmp.log"; exit 1; fi
        clang -target nvptx64-nvidia-cuda -march=sm_75 \
              -Xclang -target-feature -Xclang +ptx${ptx} \
              -O3 -ffp-contract=fast -ffreestanding \
              -Wall -Wextra \
              -S kernels/device_kernels.c -o ops.ptx 2> "$out_tmp.log" || { cat "$out_tmp.log" >&2; rm -f "$out_tmp" "$out_tmp.log"; exit 1; }
        echo "warn: using ptx$ptx for ops (ptx78 not supported)" >&2
        rm -f "$out_tmp" "$out_tmp.log"
    fi
}
sed -i 's/^\.visible \.func pg_k_/.visible .entry pg_k_/' ops.ptx
grep -q '^\.visible \.entry pg_k_map' ops.ptx || { echo "ops .entry missing" >&2; exit 1; }

xxd -i sgemm.ptx > sgemm_ptx.h
xxd -i ops.ptx > ops_ptx.h

echo "ok: sgemm.ptx ($(wc -c < sgemm.ptx) bytes), ops.ptx ($(wc -c < ops.ptx) bytes)"