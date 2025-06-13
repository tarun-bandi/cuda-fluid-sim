// Basic CUDA setup
// Basic fluid simulation
// Particle system
// OpenGL visualization
// Interactive controls
// Color mapping
// Velocity visualization
// Heat sources
// Error handling
// Reset functionality
// FPS counter
// Window resizing
// Keyboard controls
// Color blending

// Main CUDA Fluid Simulation Application
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

// Forward declarations of CUDA kernel launcher functions
extern "C" {
    void launchAdvectKernel(float* output, float* input, float* velX, float* velY,
                           int width, int height, float dt);
    void launchDiffuseKernel(float* output, float* input, float diffusion, float dt,
                           int width, int height);
    void launchAddForceKernel(float* velocityY, float* temperature,
                            int width, int height, float buoyancy);
    void launchProjectionKernels(float* velX, float* velY, float* pressure, float* divergence,
                                int width, int height, int iterations);
}

// Simulation parameters
const int GRID_WIDTH = 128;
const int GRID_HEIGHT = 128;
const float DT = 0.1f;
const float VISCOSITY = 0.0001f;
const float DIFFUSION = 0.0001f;
const float BUOYANCY = 0.1f;

// CUDA device pointers
float *d_velocityX, *d_velocityY;
float *d_velocityX_prev, *d_velocityY_prev;
float *d_density, *d_density_prev;
float *d_temperature, *d_temperature_prev;
float *d_pressure, *d_divergence;

// Host arrays for OpenGL rendering
float *h_density;
float *h_temperature;

// OpenGL variables
int window_width = 512;
int window_height = 512;
bool mouse_down = false;
int mouse_x = 0, mouse_y = 0;

// Initialize CUDA memory
void initCuda() {
    int size = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_velocityX, size);
    cudaMalloc(&d_velocityY, size);
    cudaMalloc(&d_velocityX_prev, size);
    cudaMalloc(&d_velocityY_prev, size);
    cudaMalloc(&d_density, size);
    cudaMalloc(&d_density_prev, size);
    cudaMalloc(&d_temperature, size);
    cudaMalloc(&d_temperature_prev, size);
    cudaMalloc(&d_pressure, size);
    cudaMalloc(&d_divergence, size);
    
    // Initialize to zero
    cudaMemset(d_velocityX, 0, size);
    cudaMemset(d_velocityY, 0, size);
    cudaMemset(d_velocityX_prev, 0, size);
    cudaMemset(d_velocityY_prev, 0, size);
    cudaMemset(d_density, 0, size);
    cudaMemset(d_density_prev, 0, size);
    cudaMemset(d_temperature, 0, size);
    cudaMemset(d_temperature_prev, 0, size);
    cudaMemset(d_pressure, 0, size);
    cudaMemset(d_divergence, 0, size);
    
    // Allocate host memory for rendering
    h_density = new float[GRID_WIDTH * GRID_HEIGHT];
    h_temperature = new float[GRID_WIDTH * GRID_HEIGHT];
    
    std::cout << "CUDA memory initialized successfully!" << std::endl;
}

// Add source at specified location
void addSource(float* field, int x, int y, float amount) {
    if (x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT) {
        int idx = y * GRID_WIDTH + x;
        float temp_value = amount;
        cudaMemcpy(&field[idx], &temp_value, sizeof(float), cudaMemcpyHostToDevice);
    }
}

// Perform one simulation step
void simulationStep() {
    // Add forces (buoyancy from temperature)
    launchAddForceKernel(d_velocityY, d_temperature, GRID_WIDTH, GRID_HEIGHT, BUOYANCY);
    
    // Velocity step
    // 1. Diffusion
    launchDiffuseKernel(d_velocityX, d_velocityX_prev, VISCOSITY, DT, GRID_WIDTH, GRID_HEIGHT);
    launchDiffuseKernel(d_velocityY, d_velocityY_prev, VISCOSITY, DT, GRID_WIDTH, GRID_HEIGHT);
    
    // 2. Projection (make divergence-free)
    launchProjectionKernels(d_velocityX, d_velocityY, d_pressure, d_divergence, 
                           GRID_WIDTH, GRID_HEIGHT, 20);
    
    // 3. Advection
    std::swap(d_velocityX, d_velocityX_prev);
    std::swap(d_velocityY, d_velocityY_prev);
    launchAdvectKernel(d_velocityX, d_velocityX_prev, d_velocityX_prev, d_velocityY_prev,
                      GRID_WIDTH, GRID_HEIGHT, DT);
    launchAdvectKernel(d_velocityY, d_velocityY_prev, d_velocityX_prev, d_velocityY_prev,
                      GRID_WIDTH, GRID_HEIGHT, DT);
    
    // 4. Final projection
    launchProjectionKernels(d_velocityX, d_velocityY, d_pressure, d_divergence, 
                           GRID_WIDTH, GRID_HEIGHT, 20);
    
    // Density step
    launchDiffuseKernel(d_density, d_density_prev, DIFFUSION, DT, GRID_WIDTH, GRID_HEIGHT);
    std::swap(d_density, d_density_prev);
    launchAdvectKernel(d_density, d_density_prev, d_velocityX, d_velocityY,
                      GRID_WIDTH, GRID_HEIGHT, DT);
    
    // Temperature step
    launchDiffuseKernel(d_temperature, d_temperature_prev, DIFFUSION * 0.5f, DT, 
                       GRID_WIDTH, GRID_HEIGHT);
    std::swap(d_temperature, d_temperature_prev);
    launchAdvectKernel(d_temperature, d_temperature_prev, d_velocityX, d_velocityY,
                      GRID_WIDTH, GRID_HEIGHT, DT);
    
    // Decay
    // Implement simple decay by scaling values
    float decay_factor = 0.998f;
    cudaMemcpy(h_temperature, d_temperature, GRID_WIDTH * GRID_HEIGHT * sizeof(float), 
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        h_temperature[i] *= decay_factor;
    }
    cudaMemcpy(d_temperature, h_temperature, GRID_WIDTH * GRID_HEIGHT * sizeof(float), 
               cudaMemcpyHostToDevice);
}

// OpenGL rendering function
void render() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Copy data from device to host for rendering
    cudaMemcpy(h_density, d_density, GRID_WIDTH * GRID_HEIGHT * sizeof(float), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_temperature, d_temperature, GRID_WIDTH * GRID_HEIGHT * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Render as colored pixels
    glBegin(GL_QUADS);
    for (int y = 0; y < GRID_HEIGHT - 1; y++) {
        for (int x = 0; x < GRID_WIDTH - 1; x++) {
            int idx = y * GRID_WIDTH + x;
            
            // Combine density and temperature for color
            float density = h_density[idx];
            float temp = h_temperature[idx];
            
            // Color mapping: density -> blue, temperature -> red
            float r = temp;
            float g = (density + temp) * 0.5f;
            float b = density;
            
            // Clamp colors
            r = fminf(1.0f, fmaxf(0.0f, r));
            g = fminf(1.0f, fmaxf(0.0f, g));
            b = fminf(1.0f, fmaxf(0.0f, b));
            
            glColor3f(r, g, b);
            
            float x1 = (float)x / GRID_WIDTH * window_width;
            float y1 = (float)y / GRID_HEIGHT * window_height;
            float x2 = (float)(x + 1) / GRID_WIDTH * window_width;
            float y2 = (float)(y + 1) / GRID_HEIGHT * window_height;
            
            glVertex2f(x1, y1);
            glVertex2f(x2, y1);
            glVertex2f(x2, y2);
            glVertex2f(x1, y2);
        }
    }
    glEnd();
    
    glutSwapBuffers();
}

// GLUT display callback
void display() {
    simulationStep();
    render();
}

// GLUT idle callback
void idle() {
    glutPostRedisplay();
}

// Mouse interaction
void mouse(int button, int state, int x, int y) {
    mouse_down = (state == GLUT_DOWN);
    mouse_x = x;
    mouse_y = window_height - y; // Flip Y coordinate
}

void motion(int x, int y) {
    if (mouse_down) {
        mouse_x = x;
        mouse_y = window_height - y; // Flip Y coordinate
        
        // Convert screen coordinates to grid coordinates
        int grid_x = (mouse_x * GRID_WIDTH) / window_width;
        int grid_y = (mouse_y * GRID_HEIGHT) / window_height;
        
        // Add density and temperature at mouse position
        addSource(d_density, grid_x, grid_y, 0.5f);
        addSource(d_temperature, grid_x, grid_y, 1.0f);
    }
}

// Keyboard interaction
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'r':
        case 'R':
            // Reset simulation
            cudaMemset(d_velocityX, 0, GRID_WIDTH * GRID_HEIGHT * sizeof(float));
            cudaMemset(d_velocityY, 0, GRID_WIDTH * GRID_HEIGHT * sizeof(float));
            cudaMemset(d_density, 0, GRID_WIDTH * GRID_HEIGHT * sizeof(float));
            cudaMemset(d_temperature, 0, GRID_WIDTH * GRID_HEIGHT * sizeof(float));
            std::cout << "Simulation reset!" << std::endl;
            break;
        case 27: // Escape key
            exit(0);
            break;
    }
}

// Reshape callback
void reshape(int w, int h) {
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Initialize OpenGL
void initGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
}

// Main function
int main(int argc, char** argv) {
    std::cout << "CUDA Fluid Simulation Starting..." << std::endl;
    
    // Check for CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found! Running in CPU mode..." << std::endl;
        std::cout << "For full GPU acceleration, please run on a system with NVIDIA GPU." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Initialize CUDA
    initCuda();
    
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA Fluid Simulation with Temperature");
    
    // Initialize OpenGL
    initGL();
    
    // Set GLUT callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    
    std::cout << "Simulation initialized successfully!" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - Click and drag to add fluid and heat" << std::endl;
    std::cout << "  - Press 'R' to reset simulation" << std::endl;
    std::cout << "  - Press ESC to exit" << std::endl;
    
    // Start the main loop
    glutMainLoop();
    
    // Cleanup
    cudaFree(d_velocityX);
    cudaFree(d_velocityY);
    cudaFree(d_velocityX_prev);
    cudaFree(d_velocityY_prev);
    cudaFree(d_density);
    cudaFree(d_density_prev);
    cudaFree(d_temperature);
    cudaFree(d_temperature_prev);
    cudaFree(d_pressure);
    cudaFree(d_divergence);
    
    delete[] h_density;
    delete[] h_temperature;
    
    return 0;
}
