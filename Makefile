# picograd

CC       ?= cc
AR       ?= ar

CFLAGS   ?= -O2 -g
CFLAGS   += -std=c11 -Wall -Wextra -mavx2 -fno-fast-math
CPPFLAGS += -Isrc -D_POSIX_C_SOURCE=200809L
ASFLAGS  += -mavx2

LDFLAGS  +=
LDLIBS   += -lm

# MKL for benchmarks (Intel oneAPI)
MKLROOT  ?= /opt/intel/oneapi/mkl/latest

BUILD    := build

SRC      := $(shell find src -name '*.c')
ASM      := $(shell find src -name '*.S')
OBJ      := $(patsubst %.c,$(BUILD)/%.o,$(SRC)) $(patsubst %.S,$(BUILD)/%.o,$(ASM))
DEP      := $(OBJ:.o=.d)

LIB      := $(BUILD)/libpicograd.a

TESTS    := $(patsubst tests/%.c,$(BUILD)/%,$(wildcard tests/*.c))
BENCH    := $(BUILD)/bench_gemm

.PHONY: all test bench clean

all: $(LIB) $(TESTS)

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

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; done

# benchmark (requires MKL)

$(BENCH): benchmarks/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) $(CPPFLAGS) \
	    -I$(MKLROOT)/include \
	    $< $(LIB) \
	    -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 \
	    -lmkl_rt $(LDFLAGS) $(LDLIBS) -o $@

bench: $(BENCH)
	./$(BENCH)

clean:
	rm -rf $(BUILD)

-include $(DEP)
