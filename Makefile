# picograd

CC       ?= cc
AR       ?= ar

CFLAGS   ?= -O3 -g
CFLAGS   += -std=c11 -Wall -Wextra -ffast-math -funroll-loops -ftree-vectorize -fPIC
CPPFLAGS += -Isrc -D_POSIX_C_SOURCE=200809L

# ---- arch abstraction ----
ARCH     ?= $(shell uname -m)
ifeq ($(ARCH),amd64)
override ARCH := x86_64
endif
ifeq ($(ARCH),arm64)
override ARCH := aarch64
endif
# NEOVERSE: n1 (default Neoverse-N1, dotprod only), v1 (SVE 256), v2 (SVE2)
NEOVERSE ?= n1

ifeq ($(ARCH),x86_64)
CFLAGS   += -mavx2
ASFLAGS  += -mavx2 -mavx512f
CPPFLAGS += -DPG_ARCH_X86_64
else ifeq ($(ARCH),aarch64)
ASFLAGS :=
CPPFLAGS += -DPG_ARCH_AARCH64
  # base for N1: armv8.2-a + fp16 + dotprod
  CFLAGS += -march=armv8.2-a+fp16+dotprod
  # optional SVE tunings per Neoverse
  ifeq ($(NEOVERSE),v1)
    CFLAGS += -march=armv8.4-a+fp16+dotprod+sve
    CPPFLAGS += -DPG_HAVE_SVE=1
    HAS_SVE := 1
  else ifeq ($(NEOVERSE),v2)
    CFLAGS += -march=armv9-a+sve2+bf16
    CPPFLAGS += -DPG_HAVE_SVE=1 -DPG_HAVE_SVE2=1
    HAS_SVE := 1
  else
    # probe if compiler supports SVE at all; if yes enable SVE for sve micro only
    HAS_SVE_PROBE := $(shell printf 'int main(){return 0;}' | $(CC) -march=armv8-a+sve -c -x c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
    ifeq ($(HAS_SVE_PROBE),1)
      CPPFLAGS += -DPG_HAVE_SVE=1
      HAS_SVE := 1
    else
      HAS_SVE := 0
    endif
  endif
else
# generic: no SIMD ISA flags, rely on -ftree-vectorize + scalar
ASFLAGS  :=
CPPFLAGS += -DPG_ARCH_GENERIC
HAS_SVE  := 0
endif

LDFLAGS  +=
LDLIBS   += -lm -ldl -pthread
CPPFLAGS += -pthread

# MKL for benchmarks (Intel oneAPI)
MKLROOT  ?= /opt/intel/oneapi/mkl/latest

# Intel SDE for testing AVX-512 kernels on CPUs without them
SDE      ?= $(HOME)/tools/sde/sde
SDE_CPU  ?= -spr

# GPU backend: cpu (default) | cuda | metal | hip (rocm alias)
# cuda uses CUDA Driver API via dlopen + embedded PTX kernels;
# no CUDA toolkit is needed to build or run it, only the NVIDIA driver.
# metal uses Metal.framework (Apple only); on non-Apple it builds as stub.
# hip/rocm uses HIP runtime via dlopen + hiprtc runtime compilation;
# no ROCm toolkit is needed to build, only the ROCm driver at run time.
BACKEND   ?= cpu

# rocm is an alias for hip
ifeq ($(BACKEND),rocm)
override BACKEND := hip
endif

BUILD    := build

SRC      := $(shell find src -name '*.c' -not -path 'src/backend/cuda/*' -not -path 'src/backend/metal/*' -not -path 'src/backend/hip/*')
ASM      := $(shell find src -name '*.S')

# --- arch source filtering ---
ifeq ($(ARCH),x86_64)
# x86: keep ASM, exclude ARM gemm kernels if present
SRC := $(filter-out src/backend/cpu/gemm_neon.c src/backend/cpu/gemm_sve.c src/backend/cpu/gemm_generic.c,$(SRC))
else ifeq ($(ARCH),aarch64)
ASM :=
ifeq ($(HAS_SVE),0)
SRC := $(filter-out src/backend/cpu/gemm_sve.c,$(SRC))
endif
ifeq ($(HAS_SVE),1)
# ensure sve file built with SVE when base CFLAGS lacks it (n1 probe case)
$(BUILD)/src/backend/cpu/gemm_sve.o: CFLAGS += -march=armv8-a+sve
endif
else
# generic (qemu/CI): no ASM, no neon/sve, only generic gemm
ASM :=
SRC := $(filter-out src/backend/cpu/gemm_neon.c src/backend/cpu/gemm_sve.c,$(SRC))
endif

ifeq ($(BACKEND),cuda)
CPPFLAGS += -DPICOGRAD_BACKEND_CUDA
LDLIBS   += -ldl
SRC      += $(wildcard src/backend/cuda/*.c)
endif

ifeq ($(BACKEND),metal)
CPPFLAGS += -DPICOGRAD_BACKEND_METAL
SRC      += $(wildcard src/backend/metal/*.c)
SRC      += $(wildcard src/backend/metal/*.m)
ifeq ($(shell uname),Darwin)
LDLIBS   += -framework Metal -framework Foundation -lobjc
# metal.c contains Objective-C (Metal/Foundation imports, blocks).
# Force clang to compile it as Objective-C on macOS (no ARC – avoids
# bridged-cast errors; metal uses manual buffer-pointer handling).
$(BUILD)/src/backend/metal/%.o: CFLAGS += -x objective-c -fblocks -fno-objc-arc
endif
endif

ifeq ($(BACKEND),hip)
CPPFLAGS += -DPICOGRAD_BACKEND_HIP
LDLIBS   += -ldl
SRC      += $(wildcard src/backend/hip/*.c)
endif

OBJ      := $(patsubst %.c,$(BUILD)/%.o,$(filter %.c,$(SRC))) \
            $(patsubst %.m,$(BUILD)/%.o,$(filter %.m,$(SRC))) \
            $(patsubst %.S,$(BUILD)/%.o,$(ASM))
DEP      := $(OBJ:.o=.d)

LIB      := $(BUILD)/libpicograd.a
SHARED   := $(BUILD)/libpicograd.so

TESTS    := $(patsubst tests/%.c,$(BUILD)/%,$(wildcard tests/*.c))
EXAMPLES := $(patsubst examples/%.c,$(BUILD)/%,$(wildcard examples/*.c))
BENCH    := $(BUILD)/bench_gemm
BENCH_GPU :=
ifeq ($(BACKEND),cuda)
BENCH_GPU := $(BUILD)/bench_gemm_cuda
endif
ifeq ($(BACKEND),metal)
BENCH_GPU := $(BUILD)/bench_gemm_metal
endif
ifeq ($(BACKEND),hip)
BENCH_GPU := $(BUILD)/bench_gemm_hip
endif

.PHONY: all test examples bench bench-gpu test-sde clean shared

all: $(LIB) $(SHARED) $(TESTS) $(EXAMPLES) $(BENCH_GPU)

# libraries

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHARED): $(OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

shared: $(SHARED)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD)/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -fobjc-arc $(CPPFLAGS) -MMD -MP -c $< -o $@

# tests

$(BUILD)/test_%: tests/test_%.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/%: examples/%.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

examples: $(EXAMPLES)
	@echo "examples compiled:"; for e in $(EXAMPLES); do echo "  $$e"; done

# run tests under Intel SDE (emulates AVX-512 on CPUs without it)
test-sde: $(TESTS)
	@for t in $(TESTS); do $(SDE) $(SDE_CPU) -- ./$$t || exit 1; done

# benchmark (requires MKL on x86_64, otherwise generic)
ifeq ($(ARCH),x86_64)
$(BENCH): benchmarks/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) \
	    -I$(MKLROOT)/include \
	    $< $(LIB) \
	    -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
	    -lmkl_rt $(LDFLAGS) $(LDLIBS) -o $@
else
$(BENCH): benchmarks/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@
endif

bench: $(BENCH)
	./$(BENCH)

# GPU backend benchmark (requires BACKEND=cuda|metal|hip and a GPU)

$(BUILD)/bench_gemm_cuda: benchmarks/bench_gemm_cuda.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/bench_gemm_metal: benchmarks/bench_gemm_cuda.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/bench_gemm_hip: benchmarks/bench_gemm_hip.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

bench-gpu: $(BENCH_GPU)
	./$(BENCH_GPU)

clean:
	rm -rf $(BUILD)

-include $(DEP)
