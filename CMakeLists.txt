cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(path_trace LANGUAGES CXX CUDA)
enable_language(CUDA)

SET( EIGEN3_INCLUDE_DIR "/users/bblinn/pt_inv/include/eigen3")
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

add_compile_options(-w)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_library(${PROJECT_NAME}.so SHARED
#			  path_trace.cu)
add_executable(${PROJECT_NAME} path_trace.cu)
add_executable(temp temp.cu)

#set_property(TARGET ${PROJECT_NAME}.so PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
set_property(TARGET temp PROPERTY CUDA_STANDARD 11)
