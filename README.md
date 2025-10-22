# Fast Discrete Orthonormal Stockwell Transform (FDOST) and Stockwell Transform (S-Transform)

## Overview

This repository implements both CPU and GPU versions of the Fast Discrete Orthonormal Stockwell Transform (FDOST) and the Stockwell Transform (S-Transform) for real-valued time series data.
The transforms decompose a signal into its localized time–frequency representation while preserving signal energy and (in the case of FDOST) orthonormality.

The GPU implementations use CuPy for efficient parallel computation on NVIDIA GPUs.

## Features

### **FDOST (CPU + GPU)**
- Efficient orthonormal transform for real-valued input  
- Exploits conjugate symmetry for negative frequencies  
- Vectorized matrix-based implementation for performance  
- GPU acceleration using CuPy  

### **Stockwell Transform (CPU + GPU)**
- Professor-provided vectorized version adapted for GPU  
- Computes full time–frequency representation using Gaussian windows  
- Supports frequency sampling rate control  

### **Analysis Utilities**
- Parseval’s theorem check (energy preservation)  
- Orthonormality test for FDOST basis vectors  
- Time–frequency visualization (`plot_st`)  