// Velocity field implementation
// Force application
// Temperature field
// Temperature advection
// Temperature diffusion
// Performance optimizations
// Boundary conditions
// Final optimizations

// CUDA Kernel implementations for Fluid Simulation
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

// Texture memory declarations for better performance
texture<float, 2, cudaReadModeElementType> texVelocityX;
texture<float, 2, cudaReadModeElementType> texVelocityY;
texture<float, 2, cudaReadModeElementType> texDensity;
texture<float, 2, cudaReadModeElementType> texTemperature;

// CUDA kernel for advection using backwards Euler method
__global__ void advectKernel(float* output, float* input, 
                           float* velocityX, float* velocityY,
                           int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Backwards advection - trace particle backwards in time
    float px = x - dt * velocityX[idx] * width;
    float py = y - dt * velocityY[idx] * height;
    
    // Clamp coordinates to stay within bounds
    px = fmaxf(0.5f, fminf(width - 0.5f, px));
    py = fmaxf(0.5f, fminf(height - 0.5f, py));
    
    // Bilinear interpolation for smooth sampling
    int x0 = (int)px; int x1 = x0 + 1;
    int y0 = (int)py; int y1 = y0 + 1;
    
    float fx = px - x0;
    float fy = py - y0;
    
    // Ensure we don't go out of bounds
    x1 = min(x1, width - 1);
    y1 = min(y1, height - 1);
    
    float val = (1-fx)*(1-fy)*input[y0*width + x0] +
               fx*(1-fy)*input[y0*width + x1] +
               (1-fx)*fy*input[y1*width + x0] +
               fx*fy*input[y1*width + x1];
    
    output[idx] = val;
}

// CUDA kernel for diffusion using Gauss-Seidel iteration
__global__ void diffuseKernel(float* output, float* input,
                            float diffusion, float dt,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    
    int idx = y * width + x;
    float a = dt * diffusion * width * height;
    
    output[idx] = (input[idx] + a * (
        input[(y-1)*width + x] + input[(y+1)*width + x] +
        input[y*width + (x-1)] + input[y*width + (x+1)]
    )) / (1 + 4*a);
}

// CUDA kernel for adding forces (buoyancy, external forces)
__global__ void addForceKernel(float* velocityY, float* temperature,
                             int width, int height, float buoyancy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    // Buoyancy force - hot air rises
    velocityY[idx] += temperature[idx] * buoyancy;
}

// CUDA kernel for computing divergence of velocity field
__global__ void divergenceKernel(float* divergence, float* velocityX, float* velocityY,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    
    int idx = y * width + x;
    
    divergence[idx] = -0.5f * (
        velocityX[(y)*width + (x+1)] - velocityX[(y)*width + (x-1)] +
        velocityY[(y+1)*width + (x)] - velocityY[(y-1)*width + (x)]
    ) / width;
}

// CUDA kernel for pressure projection (Poisson solve iteration)
__global__ void pressureKernel(float* pressure, float* divergence,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    
    int idx = y * width + x;
    
    pressure[idx] = (divergence[idx] + 
        pressure[(y-1)*width + x] + pressure[(y+1)*width + x] +
        pressure[y*width + (x-1)] + pressure[y*width + (x+1)]
    ) / 4.0f;
}

// CUDA kernel for subtracting pressure gradient
__global__ void gradientSubtractionKernel(float* velocityX, float* velocityY,
                                        float* pressure, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    
    int idx = y * width + x;
    
    velocityX[idx] -= 0.5f * width * (pressure[y*width + (x+1)] - pressure[y*width + (x-1)]);
    velocityY[idx] -= 0.5f * width * (pressure[(y+1)*width + x] - pressure[(y-1)*width + x]);
}

// CUDA kernel for setting boundary conditions
__global__ void setBoundaryKernel(float* field, int width, int height, int boundary_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= 2 * (width + height)) return;
    
    if (idx < width) {
        // Top boundary
        int x = idx;
        if (boundary_type == 1) field[0 * width + x] = -field[1 * width + x]; // x-velocity
        else if (boundary_type == 2) field[0 * width + x] = field[1 * width + x]; // y-velocity
        else field[0 * width + x] = field[1 * width + x]; // density/temperature
    } else if (idx < 2 * width) {
        // Bottom boundary
        int x = idx - width;
        if (boundary_type == 1) field[(height-1) * width + x] = -field[(height-2) * width + x];
        else if (boundary_type == 2) field[(height-1) * width + x] = field[(height-2) * width + x];
        else field[(height-1) * width + x] = field[(height-2) * width + x];
    } else if (idx < 2 * width + height) {
        // Left boundary
        int y = idx - 2 * width;
        if (boundary_type == 1) field[y * width + 0] = field[y * width + 1];
        else if (boundary_type == 2) field[y * width + 0] = -field[y * width + 1];
        else field[y * width + 0] = field[y * width + 1];
    } else {
        // Right boundary
        int y = idx - 2 * width - height;
        if (boundary_type == 1) field[y * width + (width-1)] = field[y * width + (width-2)];
        else if (boundary_type == 2) field[y * width + (width-1)] = -field[y * width + (width-2)];
        else field[y * width + (width-1)] = field[y * width + (width-2)];
    }
}

// CUDA kernel for adding density/temperature sources
__global__ void addSourceKernel(float* field, float* source, 
                              int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    field[idx] += dt * source[idx];
}

// CUDA kernel for vorticity confinement (adds interesting swirls)
__global__ void vorticityKernel(float* vorticity, float* velocityX, float* velocityY,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    
    int idx = y * width + x;
    
    vorticity[idx] = 0.5f * (
        velocityX[(y+1)*width + x] - velocityX[(y-1)*width + x] -
        velocityY[y*width + (x+1)] + velocityY[y*width + (x-1)]
    );
}

// Host function to launch kernels with proper grid/block dimensions
extern "C" {
    void launchAdvectKernel(float* output, float* input, float* velX, float* velY,
                           int width, int height, float dt) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        advectKernel<<<gridSize, blockSize>>>(output, input, velX, velY, width, height, dt);
        cudaDeviceSynchronize();
    }
    
    void launchDiffuseKernel(float* output, float* input, float diffusion, float dt,
                           int width, int height) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        diffuseKernel<<<gridSize, blockSize>>>(output, input, diffusion, dt, width, height);
        cudaDeviceSynchronize();
    }
    
    void launchAddForceKernel(float* velocityY, float* temperature,
                            int width, int height, float buoyancy) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        addForceKernel<<<gridSize, blockSize>>>(velocityY, temperature, width, height, buoyancy);
        cudaDeviceSynchronize();
    }
    
    void launchProjectionKernels(float* velX, float* velY, float* pressure, float* divergence,
                                int width, int height, int iterations) {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Compute divergence
        divergenceKernel<<<gridSize, blockSize>>>(divergence, velX, velY, width, height);
        cudaDeviceSynchronize();
        
        // Solve Poisson equation iteratively
        for (int i = 0; i < iterations; i++) {
            pressureKernel<<<gridSize, blockSize>>>(pressure, divergence, width, height);
            cudaDeviceSynchronize();
        }
        
        // Subtract pressure gradient
        gradientSubtractionKernel<<<gridSize, blockSize>>>(velX, velY, pressure, width, height);
        cudaDeviceSynchronize();
    }
}
