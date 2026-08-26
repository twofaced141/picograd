# picograd

CC       ?= cc
AR       ?= ar

CFLAGS   ?= -O2 -g
CFLAGS   += -std=c11 -Wall -Wextra -mavx2 -fno-fast-math
CPPFLAGS += -Isrc -D_POSIX_C_SOURCE=200809L
ASFLAGS  += -mavx2 -mavx512f

LDFLAGS  +=
LDLIBS   += -lm

# MKL for benchmarks (Intel oneAPI)
MKLROOT  ?= /opt/intel/oneapi/mkl/latest

# Intel SDE for testing AVX-512 kernels on CPUs without them
SDE      ?= $(HOME)/tools/sde/sde
SDE_CPU  ?= -spr

# GPU backend: cpu (default) | cuda
# cuda uses CUDA Driver API via dlopen + embedded PTX kernels;
# no CUDA toolkit is needed to build or run it, only the NVIDIA driver.
BACKEND   ?= cpu

BUILD    := build

SRC      := $(shell find src -name '*.c' -not -path 'src/backend/cuda/*')
ASM      := $(shell find src -name '*.S')

ifeq ($(BACKEND),cuda)
CPPFLAGS += -DPICOGRAD_BACKEND_CUDA
LDLIBS   += -ldl
SRC      += $(wildcard src/backend/cuda/*.c)
endif

OBJ      := $(patsubst %.c,$(BUILD)/%.o,$(SRC)) $(patsubst %.S,$(BUILD)/%.o,$(ASM))
DEP      := $(OBJ:.o=.d)

LIB      := $(BUILD)/libpicograd.a

TESTS    := $(patsubst tests/%.c,$(BUILD)/%,$(wildcard tests/*.c))
EXAMPLES := $(patsubst examples/%.c,$(BUILD)/%,$(wildcard examples/*.c))
BENCH    := $(BUILD)/bench_gemm
BENCH_GPU :=
ifeq ($(BACKEND),cuda)
BENCH_GPU := $(BUILD)/bench_gemm_cuda
endif

.PHONY: all test examples bench bench-gpu test-sde clean

all: $(LIB) $(TESTS) $(EXAMPLES) $(BENCH_GPU)

# library

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

# tests

$(BUILD)/test_%: tests/test_%.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD)/%: examples/%.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

examples: $(EXAMPLES)
	@for e in $(EXAMPLES); do echo "== $$e =="; ./$$e || exit 1; done

# run tests under Intel SDE (emulates AVX-512 on CPUs without it)
test-sde: $(TESTS)
	@for t in $(TESTS); do $(SDE) $(SDE_CPU) -- ./$$t || exit 1; done

# benchmark (requires MKL)

$(BENCH): benchmarks/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) \
	    -I$(MKLROOT)/include \
	    $< $(LIB) \
	    -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
	    -lmkl_rt $(LDFLAGS) $(LDLIBS) -o $@

bench: $(BENCH)
	./$(BENCH)

# GPU backend benchmark (requires BACKEND=cuda and a CUDA GPU)

$(BENCH_GPU): benchmarks/bench_gemm_cuda.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

bench-gpu: $(BENCH_GPU)
	./$(BENCH_GPU)

clean:
	rm -rf $(BUILD)

-include $(DEP)
