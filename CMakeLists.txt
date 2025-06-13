cmake_minimum_required(VERSION 3.10)
project(cuda_fluid_sim LANGUAGES CUDA CXX)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Set CUDA standard and compatibility
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture flags for better compatibility
# Use multiple architectures to support different GPUs
set(CMAKE_CUDA_ARCHITECTURES "35;50;60;70;75")

# Alternative approach for older CMake versions
if(CMAKE_VERSION VERSION_LESS "3.18")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50") 
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
    # Add PTX for forward compatibility
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=compute_75")
endif()

# Add compiler flags for compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})

# Create executable
if(EXISTS "${CMAKE_SOURCE_DIR}/src/fluidsGL.cpp")
    add_executable(fluidsGL
        src/fluidsGL.cpp
        src/fluidsGL_kernels.cu
    )
    
    # Set CUDA properties
    set_property(TARGET fluidsGL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET fluidsGL PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Link libraries
    target_link_libraries(fluidsGL
        ${CUDA_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARY}
        cufft
    )
else()
    # Create a simple test file if main doesn't exist
    file(WRITE "${CMAKE_SOURCE_DIR}/src/simple_test.cpp"
        "#include <iostream>\n"
        "#include <cuda_runtime.h>\n"
        "\nint main() {\n"
        "    std::cout << \"CUDA Fluid Simulation!\\n\";\n"
        "    int deviceCount;\n"
        "    cudaGetDeviceCount(&deviceCount);\n"
        "    std::cout << \"CUDA devices found: \" << deviceCount << std::endl;\n"
        "    return 0;\n"
        "}\n")
    
    add_executable(simple_test
        src/simple_test.cpp
    )
    
    target_link_libraries(simple_test ${CUDA_LIBRARIES})
endif()

# Add a simple CUDA test executable
add_executable(cuda_compatibility_test
    src/fluidsGL_kernels.cu
)

set_property(TARGET cuda_compatibility_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_compatibility_test ${CUDA_LIBRARIES})
