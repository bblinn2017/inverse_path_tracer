cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(path_trace LANGUAGES CXX CUDA)
enable_language(CUDA)

SET( EIGEN3_INCLUDE_DIR "/users/bblinn/pt_inv/include/eigen3")
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

add_compile_options(-w)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_library(${PROJECT_NAME}.so SHARED
#			  path_trace.cu)
add_executable(${PROJECT_NAME} path_trace.cu)

#set_property(TARGET ${PROJECT_NAME}.so PROPERTY CUDA_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
