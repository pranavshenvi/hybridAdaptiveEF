# Hybrid Architecture Adaptive HNSW (withComparisons Branch)

This repository contains the implementation and benchmarking suite for the **Hybrid Architecture Adaptive HNSW**, a novel routing methodology that improves upon the Global Ada-EF algorithm by utilizing soft-blended local anchor statistics to dynamically predict search budgets (`ef`) in highly-skewed, low-dimensional vector spaces.

### 🌟 Branch: `withComparisons`
This branch merges the latest core engine updates from your team with new **multimodal comparisons**. 
1. **The `chao_hybrid_ada_ef` Engine:** The core C++ engine and Python bindings have been moved into their own dedicated library folder with extensive tests and examples.
2. **Extensive K-Value Experiments:** We conducted exhaustive sweeps across a wide range of `k` values (number of nearest neighbors) to deeply analyze routing performance, clustering impacts, and pareto-optimal frontiers using our new scripts (`benchmark_k_sweep.py`, `benchmark_pareto.py`, etc.).
3. **Audio & Image Modalities:** To prove the robust performance of our architecture across different domains, we have added new benchmarking scripts for Audio and Image embeddings, comparing our approach directly against the Exact Ada-ef Paper sweep.

## 📁 Repository Structure

### 1. Core Engine (`chao_hybrid_ada_ef/`)
This directory contains the entire C++ implementation and Python bindings for the Hybrid Architecture.
*   `hnswlib/`: The core C++ headers (`adaptive_ef.h`, `hnswalg.h`, `bruteforce.h`).
*   `python_bindings/`: The `pybind11` wrappers that expose the C++ engine to Python.
*   `tests/` & `examples/`: Comprehensive testing and usage examples.

### 2. Multimodal Benchmarking Suite (NEW)
*   **Audio (ESC50):** `benchmark_esc50_new.py` - Benchmarks 2,000 samples of 512d audio embeddings.
*   **Audio (Spotify):** `benchmark_spotify_new.py` - Benchmarks 490K Spotify track embeddings.
*   **Image (Deep Image):** `benchmark_deep_image_new.py` - Benchmarks 1 Million Deep Image vectors.

### 3. Advanced Analysis Scripts
*   `benchmark_pareto.py`: Analyzes the pareto frontier of recall vs search time (generates `pareto_front.png` and `k_metrics_dashboard.png` visualizations in the results folder).
*   `benchmark_k_sweep.py` & `benchmark_cluster_sweep.py`: Analyzes how `k` and cluster sizes affect routing performance.

### 4. Original Benchmarking Suite
*   `benchmark_skewed.py`: Helper functions for ground truth computation and synthetic skewed dataset generation.
*   `benchmark_final.py`: The executable benchmark for the Synthetic Zipfian dataset.
*   `benchmark_real.py`: The executable benchmark for the real-world MS MARCO (1M corpus) dataset.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.x installed along with the following packages:
```bash
pip install numpy scipy scikit-learn h5py matplotlib pybind11
```

### 2. Required Datasets
Because large dataset files cannot be pushed to GitHub, you must manually download and place the following files into the root directory before running the benchmarks:
*   `msmarco-1M.hdf5` (1 Million vector corpus)
*   `msmarco-8.8M-minilm-384d.hdf5` (8.8 Million vector corpus for tuning)
*   `msmarco_qemb_train.npz` (Training queries for polynomial calibration)
*   `msmarco_qemb_validation.npz` (Test queries)
*   *(For Multimodal)*:  You should also have the respective `.h5` / `.hdf5` / `.npz` data files for ESC50, Spotify, and Deep Image locally.

### 3. Compile the C++ Extension
Before running any Python scripts, you shud compile the native C++ engine. 
```bash
cd chao_hybrid_ada_ef
python setup.py build_ext --inplace
cd ..
```
*Note: This requires a C++ compiler installed on your system (e.g., MSVC on Windows, GCC/Clang on Linux).*

---

## 🏃 Running the Experiments

To run the comparisons, simply execute the respective python scripts from the root directory. The scripts will automatically handle the HNSW graph indexing, compute the ground truths, and output the sweep results comparing Ada-EF to our Hybrid approach.

```bash
# MS MARCO Real-world Benchmark (8.8M Vectors)
python benchmark_exact_paper_sweep.py

# Image Benchmark (1M Vectors)
python benchmark_deep_image_new.py

# Audio Benchmark - Spotify (490K Vectors)
python benchmark_spotify_new.py

# Audio Benchmark - ESC50 (2K Vectors)
python benchmark_esc50_new.py

# K-Value Sweeps & Pareto Analysis
python benchmark_k_sweep.py
python benchmark_pareto.py
```

*(Outputs are saved in timestamped `results_*` directories, containing detailed sweep logs, EF tables, and JSON metrics).*
