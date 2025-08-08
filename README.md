# 🖼️ GPU-Accelerated Image Filtering with CUDA

## 🔍 Overview

This project showcases how GPU acceleration can drastically improve performance for image processing tasks. Using **CUDA**, it applies a **Gaussian Blur** filter on grayscale images + **Sobel Filter** and achieves a **59x–61x speedup** over a CPU-based implementation.

The pipeline includes:
1. **Converting colour images to grayscale**
2. **Applying Gaussian blur on CPU and GPU**
3. **Applying Sobel blur on CPU and GPU**

## 🚀 Features

- Fast Gaussian and Sobel filters using CUDA
- Baseline comparison using CPU
- Grayscale conversion built-in
- Performance benchmarking
- Visual output comparison

## ⚙️ Technologies Used

| Technology | Role |
|------------|------|
| **CUDA** | Parallelized image filtering |
| **C++** | Image I/O and CPU processing |
| **Chrono** | Timing and benchmarking |
| **PGM/PPM** | Lightweight image formats for simplicity |
| **Nsight Compute** | Prolfiling tool to analyze perf. |

## 💻 Hardware Used

| Component | Specs |
|-----------|-------|
| **GPU** | NVIDIA RTX 5060 Ti (16GB) |
| **CPU** | AMD Ryzen 9 5950X (16-core) |

## 📊 Performance

| Method | Avg Time | Speedup |
|--------|----------|---------|
| **CPU** | `6.31 ms` | 1x |
| **GPU** | `0.107 ms` | **59x–61x** |

> Performance gain is due to the GPU’s ability to run thousands of threads in parallel — ideal for image convolutions.

## 📸 Example Results

| Input | Greyscale | Gaussian Blur | Sobel Filter |
|-------|------------|------------|-------------|
| <img width="509" height="512" alt="image" src="https://github.com/user-attachments/assets/e70690a6-db01-45ea-a76e-19069bafacc6" /> |  <img width="511" height="513" alt="image" src="https://github.com/user-attachments/assets/338ab95f-f188-478b-8334-d4f6d0be4444" /> | <img width="511" height="513" alt="image" src="https://github.com/user-attachments/assets/9c211b1c-a180-4959-aaf2-57cac69f20a8" /> | <img width="510" height="510" alt="image" src="https://github.com/user-attachments/assets/950e8193-6a4b-4750-aeed-dfc5904974f4" /> |


## 📌 Improvements

- Improved Profiling and optimazations using Nsight Compute 
- Multi-channel (RGB) support
