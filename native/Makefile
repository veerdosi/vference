PYTHON ?= ../.venv/bin/python
PY_SITE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
NANOBIND_DIR := $(shell $(PYTHON) -m nanobind --cmake_dir)
MLX_CMAKE_DIR := $(PY_SITE)/mlx/share/cmake/MLX
OUTPUT_DIR := $(abspath ../src/vference/native)
SDKROOT := $(shell xcrun --sdk macosx --show-sdk-path)
CXX := $(shell xcrun --find clang++)
BUILD_DIR := build-macos

.PHONY: build clean

build:
	$(PYTHON) -m cmake -S . -B $(BUILD_DIR) \
		-DPython_EXECUTABLE=$(abspath $(PYTHON)) \
		-Dnanobind_DIR=$(NANOBIND_DIR) \
		-DMLX_DIR=$(MLX_CMAKE_DIR) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CXX_FLAGS=-isystem$(SDKROOT)/usr/include/c++/v1 \
		-DCMAKE_OSX_SYSROOT=$(SDKROOT) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(OUTPUT_DIR)
	$(PYTHON) -m cmake --build $(BUILD_DIR) --target _vference_native -j

clean:
	$(PYTHON) -m cmake --build $(BUILD_DIR) --target clean
