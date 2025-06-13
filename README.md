# CUDA Fluid Simulation with Temperature Visualization

A real-time 2D fluid simulation implemented in CUDA with temperature visualization and velocity-based coloring. This project demonstrates the use of CUDA for physics simulation and OpenGL for visualization.

## Features

- Real-time 2D fluid simulation using CUDA
- Temperature field visualization (blue=cold, red=hot)
- Velocity-based coloring
- Interactive force application with mouse
- Heat source addition on click
- Particle-based visualization

## Requirements

- CUDA Toolkit (11.0 or later)
- OpenGL
- GLUT/FreeGLUT
- CMake (3.10 or later)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./fluidsGL
```

## Controls

- Left click and drag to add forces
- Click to add heat sources
- 'r' to reset the simulation
- ESC to exit

## Implementation Details

The simulation uses:
- CUDA for parallel computation
- CUFFT for solving the Navier-Stokes equations
- OpenGL for visualization
- Temperature advection and diffusion
- Velocity-based coloring

## License

This project is licensed under the MIT License - see the LICENSE file for details. 