#!/bin/bash

echo "🔧 CUDA Build Test Script"
echo "========================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ CMakeLists.txt not found! Make sure you're in the project root."
    exit 1
fi

# Check for CUDA
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA compiler found:"
    nvcc --version
else
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "   On Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "   On macOS: Install CUDA toolkit from NVIDIA"
    echo "   On Windows: Install CUDA toolkit from NVIDIA"
fi

# Check for GPU
echo ""
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found - you may not have NVIDIA GPU or drivers"
    echo "   The code will still compile but won't run efficiently"
fi

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."

# Check for OpenGL/GLUT
if pkg-config --exists glut; then
    echo "✅ GLUT found"
elif [ -f "/usr/include/GL/glut.h" ] || [ -f "/System/Library/Frameworks/GLUT.framework/Headers/glut.h" ]; then
    echo "✅ GLUT headers found"
else
    echo "❌ GLUT not found!"
    echo "   On Ubuntu/Debian: sudo apt install freeglut3-dev"
    echo "   On macOS: Should be available (uses GLUT framework)"
fi

# Check for CMake
if command -v cmake &> /dev/null; then
    echo "✅ CMake found: $(cmake --version | head -1)"
else
    echo "❌ CMake not found!"
    echo "   Install CMake from https://cmake.org/"
fi

echo ""
echo "🔨 Attempting to build..."

# Create build directory
mkdir -p build
cd build

# Configure
echo "📋 Configuring with CMake..."
if cmake .. 2>&1 | tee cmake_output.log; then
    echo "✅ CMake configuration successful"
else
    echo "❌ CMake configuration failed!"
    echo "Check cmake_output.log for details"
    exit 1
fi

# Build
echo ""
echo "🔨 Building..."
if make -j$(nproc 2>/dev/null || echo 4) 2>&1 | tee build_output.log; then
    echo "✅ Build successful!"
    
    # Check if executable exists
    if [ -f "fluidsGL" ]; then
        echo "✅ Executable created: fluidsGL"
        ls -la fluidsGL
        
        # Try to run it (will fail without display in headless mode)
        echo ""
        echo "🚀 Testing executable..."
        if [ "$DISPLAY" ]; then
            echo "Display detected, trying to run..."
            timeout 5s ./fluidsGL || echo "Note: GUI app may not run in this environment"
        else
            echo "No display - executable exists but needs GUI environment to run"
            echo "To run locally: cd build && ./fluidsGL"
        fi
    else
        echo "❌ Executable not found after build!"
        ls -la
    fi
else
    echo "❌ Build failed!"
    echo "Check build_output.log for details"
    
    # Show some common issues
    echo ""
    echo "Common build issues:"
    echo "1. Missing CUDA: Install nvidia-cuda-toolkit"
    echo "2. Missing OpenGL/GLUT: Install freeglut3-dev"
    echo "3. Wrong CMake version: Need CMake 3.10+"
    echo "4. Missing C++ compiler: Install build-essential"
fi

echo ""
echo "🎯 To run the simulation locally:"
echo "   cd build && ./fluidsGL"
echo ""
echo "🎮 Controls:"
echo "   - Click and drag: Add fluid and heat"
echo "   - R: Reset simulation"
echo "   - ESC: Exit" 