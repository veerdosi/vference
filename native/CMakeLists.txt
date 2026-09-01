cmake_minimum_required(VERSION 3.22)
project(vference_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

nanobind_add_module(
  _vference_native
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  extension.cpp
)

target_link_libraries(_vference_native PRIVATE mlx ZLIB::ZLIB)

if(BUILD_SHARED_LIBS)
  target_link_options(_vference_native PRIVATE -Wl,-rpath,@loader_path)
endif()
