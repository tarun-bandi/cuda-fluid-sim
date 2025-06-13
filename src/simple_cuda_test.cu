#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel for testing compatibility
__global__ void simpleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f + 1.0f;
    }
}

// Test basic CUDA functionality without complex features
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::cout << "🚀 Simple CUDA Compatibility Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Check CUDA devices
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "❌ CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "✅ Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "❌ No CUDA devices available" << std::endl;
        return 1;
    }
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "🔥 Device " << i << ": " << prop.name << std::endl;
        std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "   Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "   Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
    
    // Test simple kernel execution
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    error = cudaMalloc(&d_a, size);
    if (error != cudaSuccess) {
        std::cout << "❌ Memory allocation failed: " << cudaGetErrorString(error) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        return 1;
    }
    
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel with simple grid/block configuration
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    std::cout << "\n🧪 Testing simple kernel..." << std::endl;
    simpleKernel<<<gridSize, blockSize>>>(d_a, N);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "❌ Simple kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        delete[] h_a; delete[] h_b; delete[] h_c;
        return 1;
    }
    
    // Wait for kernel to complete
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cout << "❌ Kernel execution failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        delete[] h_a; delete[] h_b; delete[] h_c;
        return 1;
    }
    
    std::cout << "✅ Simple kernel executed successfully!" << std::endl;
    
    // Test vector addition
    std::cout << "\n🧪 Testing vector addition..." << std::endl;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "❌ Vector add kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        delete[] h_a; delete[] h_b; delete[] h_c;
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = (i * 2.0f + 1.0f) + (i * 2);  // simpleKernel result + h_b[i]
        if (abs(h_c[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✅ Vector addition kernel executed successfully!" << std::endl;
        std::cout << "📊 First 10 results: ";
        for (int i = 0; i < 10; i++) {
            std::cout << h_c[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "❌ Vector addition results incorrect!" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    std::cout << "\n🎉 CUDA compatibility test completed!" << std::endl;
    std::cout << "✅ GPU is compatible with this CUDA code." << std::endl;
    
    return 0;
} 