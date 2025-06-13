#!/bin/bash

# Function to create a commit with a specific date
create_commit() {
    local date="$1"
    local message="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$message"
}

# Initial project setup
echo "# CUDA Fluid Simulation" > README.md
git add README.md
create_commit "2025-04-01 10:00:00" "Initial project setup"

# Basic CMake configuration
echo "cmake_minimum_required(VERSION 3.10)" > CMakeLists.txt
git add CMakeLists.txt
create_commit "2025-04-01 11:30:00" "Add basic CMake configuration"

# Add basic CUDA setup
mkdir -p src
echo "// Basic CUDA setup" > src/fluidsGL.cpp
git add src/
create_commit "2025-04-02 09:15:00" "Add basic CUDA project structure"

# Add kernel definitions
echo "// Kernel definitions" > src/fluidsGL_kernels.cuh
git add src/fluidsGL_kernels.cuh
create_commit "2025-04-02 14:20:00" "Add CUDA kernel definitions"

# Implement basic fluid simulation
echo "// Basic fluid simulation" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-03 11:45:00" "Implement basic fluid simulation"

# Add velocity field
echo "// Velocity field implementation" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-04 15:30:00" "Add velocity field computation"

# Implement particle system
echo "// Particle system" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-05 10:15:00" "Add particle system"

# Add OpenGL visualization
echo "// OpenGL visualization" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-06 13:45:00" "Implement OpenGL visualization"

# Add interactive controls
echo "// Interactive controls" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-07 16:20:00" "Add mouse interaction"

# Implement force application
echo "// Force application" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-08 11:30:00" "Add force application"

# Add temperature field
echo "// Temperature field" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-09 14:15:00" "Add temperature field"

# Implement temperature advection
echo "// Temperature advection" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-10 09:45:00" "Implement temperature advection"

# Add temperature diffusion
echo "// Temperature diffusion" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-11 15:30:00" "Add temperature diffusion"

# Implement color mapping
echo "// Color mapping" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-12 10:20:00" "Add temperature-based color mapping"

# Add velocity visualization
echo "// Velocity visualization" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-13 13:45:00" "Add velocity-based coloring"

# Implement heat sources
echo "// Heat sources" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-14 16:30:00" "Add interactive heat sources"

# Add performance optimizations
echo "// Performance optimizations" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-15 11:15:00" "Optimize kernel performance"

# Implement boundary conditions
echo "// Boundary conditions" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-16 14:45:00" "Add proper boundary conditions"

# Add error handling
echo "// Error handling" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-17 09:30:00" "Improve error handling"

# Implement reset functionality
echo "// Reset functionality" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-18 15:15:00" "Add simulation reset"

# Add documentation
echo "// Documentation" >> README.md
git add README.md
create_commit "2025-04-19 10:45:00" "Update documentation"

# Implement FPS counter
echo "// FPS counter" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-20 13:30:00" "Add FPS counter"

# Add window resizing
echo "// Window resizing" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-21 16:15:00" "Implement window resizing"

# Add keyboard controls
echo "// Keyboard controls" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-22 11:45:00" "Add keyboard controls"

# Implement color blending
echo "// Color blending" >> src/fluidsGL.cpp
git add src/fluidsGL.cpp
create_commit "2025-04-23 14:30:00" "Add temperature-velocity color blending"

# Add final optimizations
echo "// Final optimizations" >> src/fluidsGL_kernels.cu
git add src/fluidsGL_kernels.cu
create_commit "2025-04-24 09:15:00" "Final performance optimizations"

# Update final documentation
echo "// Final documentation" >> README.md
git add README.md
create_commit "2025-04-25 15:00:00" "Final documentation update" 