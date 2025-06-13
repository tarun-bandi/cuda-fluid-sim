cmake_minimum_required(VERSION 3.10)
project(cuda_fluid_sim CUDA CXX)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    fluidsGL.cpp
    fluidsGL_kernels.cu
)

# Create executable
add_executable(fluidsGL ${SOURCES})

# Include directories
target_include_directories(fluidsGL PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(fluidsGL PRIVATE
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    cufft
)

# Set CUDA specific properties
set_target_properties(fluidsGL PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
) 