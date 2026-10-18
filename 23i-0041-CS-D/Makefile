# Muhammad Ayaan Naveed | 23i-0041 | CS-D | Assignment 2
# This required conventional Makefile is the sole filename-format exception.
CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wno-unused-function -fopenmp -mavx2 -mfma -ffp-contract=off
BUILD ?= /tmp/23i-0041-CS-D-build
GB_PREFIX ?= /usr
GB_LIBDIR ?= $(GB_PREFIX)/lib/x86_64-linux-gnu
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMMON := $(ROOT)23i-0041-CS-D-TASK0-V1.h
.PHONY: all clean check
all: $(addprefix $(BUILD)/task,1 2 3 4 5)
$(BUILD):
	mkdir -p $@
define TASK_RULE
$(BUILD)/task$(1): $(ROOT)Task$(1)/23i-0041-CS-D-TASK$(1)-V1.c $(COMMON) | $(BUILD)
	$(CC) $(CFLAGS) -fopt-info-vec-optimized=$(BUILD)/task$(1).vec $$< -o $$@ -lm
endef
$(foreach n,1 2 3 4,$(eval $(call TASK_RULE,$(n))))
$(BUILD)/task5: $(ROOT)Task5/23i-0041-CS-D-TASK5-V1.c $(COMMON) | $(BUILD)
	$(CC) $(CFLAGS) -I$(GB_PREFIX)/include $< -o $@ -L$(GB_LIBDIR) -Wl,-rpath,$(GB_LIBDIR) -lgraphblas -lm
check: all
	$(BUILD)/task1 65536 4 12 static
	$(BUILD)/task2 g6 4 static
	$(BUILD)/task3 4 32
	$(BUILD)/task4 64 100 4 scalar
	$(BUILD)/task4 64 100 4 persistent
	$(BUILD)/task5 g6 4 0
clean:
	rm -f $(addprefix $(BUILD)/task,1 2 3 4 5) $(BUILD)/task*.vec
