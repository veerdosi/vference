cmake_minimum_required(VERSION 3.22)

if(APPLE AND NOT CMAKE_OSX_SYSROOT)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

if(APPLE)
  execute_process(
    COMMAND xcrun --find clang++
    OUTPUT_VARIABLE VFERENCE_APPLE_CLANGXX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(CMAKE_CXX_COMPILER "${VFERENCE_APPLE_CLANGXX}" CACHE FILEPATH "" FORCE)
endif()

project(vference_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
  include_directories(SYSTEM "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(NOT MLX_DIR)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('mlx.core').origin).parent / 'share/cmake/MLX')"
    OUTPUT_VARIABLE MLX_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()
find_package(MLX CONFIG REQUIRED)

nanobind_add_module(
  _vference_native
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  extension.cpp
)

target_link_libraries(_vference_native PRIVATE mlx)

# The MLX wheel installs libmlx beside its Python extension. Avoid embedding the
# isolated build environment's absolute path in a distributable wheel.
set_target_properties(
  _vference_native
  PROPERTIES INSTALL_RPATH "@loader_path/../../mlx/lib"
)

install(TARGETS _vference_native LIBRARY DESTINATION vference/native)

if(BUILD_SHARED_LIBS)
  target_link_options(_vference_native PRIVATE -Wl,-rpath,@loader_path)
endif()
