# Fast Discrete Orthonormal Stockwell Transform (FDOST) and Stockwell Transform (S-Transform)

## Overview

This repository implements both CPU and GPU versions of the Fast Discrete Orthonormal Stockwell Transform (FDOST) and the Stockwell Transform (S-Transform) for real-valued time series data.  
Both transforms provide a time–frequency representation of a signal, with FDOST offering an orthonormal, energy-preserving alternative to the classic S-Transform.

The GPU implementations use **CuPy** for efficient parallel computation on NVIDIA GPUs.

---

## Features

### **FDOST (CPU + GPU)**
- Efficient orthonormal transform for real-valued input  
- Exploits conjugate symmetry for negative frequencies  
- Vectorized matrix-based implementation for performance  
- GPU acceleration using CuPy  
- `fdost2m()` — converts FDOST coefficient vector into matrix dyadic band structure  

### **Stockwell Transform (CPU + GPU)**
- Professor-provided vectorized version adapted for GPU  
- Computes full time–frequency representation using Gaussian windows  
- Supports frequency sampling-rate control  
- Compatible with both NumPy (CPU) and CuPy (GPU)

---

## Analysis Utilities
- **Parseval energy check** — verifies energy preservation between time and transform domains  
- **Orthonormality test** — validates FDOST basis set  
- **`plot_st()`** — time–frequency visualization for the S-Transform  
- **`plot_fdost()`** — dyadic block visualization for FDOST (log-frequency axis optional)

---

