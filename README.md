# Hybrid Architecture Adaptive HNSW

This repository contains the implementation and benchmarking suite for the **Hybrid Architecture Adaptive HNSW**, a novel routing methodology that improves upon the Global Ada-EF algorithm by utilizing soft-blended local anchor statistics to dynamically predict search budgets (`ef`) in highly-skewed, low-dimensional vector spaces.

## 📁 Required Files to Share

To share this project with your teammates, you only need to provide the following essential files. You can ignore the rest of the temporary scratch scripts in the directory.

### Core C++ Engine
*   `cpp/adaptive_hnsw.h`: The core C++ implementation of the Hybrid Architecture, featuring layer-specific routing and dynamic polynomial `ef` evaluation.
*   `bindings.cpp`: The `pybind11` wrappers that expose the C++ engine to Python.
*   `setup.py`: The build script to compile the C++ extension.

### Python Benchmarking Suite
*   `benchmark_skewed.py`: Contains the helper functions for ground truth computation and synthetic skewed dataset generation.
*   `benchmark_final.py`: The executable benchmark for the Synthetic Zipfian dataset (proving global assumption failures).
*   `benchmark_real.py`: The executable benchmark for the real-world MS MARCO (1M corpus) dataset.
*   `plot_metrics.py`: Analyzes the JSON output from the MS MARCO benchmark and generates visualization plots.

### Required Datasets (For `benchmark_real.py`)
*   `msmarco-1M.hdf5` (The 1 Million vector corpus)
*   `msmarco_qemb_train.npz` (Training queries for polynomial calibration)
*   `msmarco_qemb_validation.npz` (Test queries)

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.x installed along with the following packages:
```bash
pip install numpy scipy scikit-learn h5py matplotlib pybind11
```

### 2. Compile the C++ Extension
Before running any Python scripts, you **must** compile the native C++ engine. Open your terminal in this directory and run:
```bash
python setup.py build_ext --inplace
```
*Note: This requires a C++ compiler installed on your system (e.g., MSVC on Windows, GCC/Clang on Linux).*

---

## 🏃 Running the Experiments

### Experiment 1: The Synthetic Zipfian Benchmark
This benchmark generates a highly skewed synthetic dataset to demonstrate the catastrophic failure mode of Global Ada-EF, and how Hybrid Architecture fixes it.
```bash
python benchmark_final.py
```

### Experiment 2: The MS MARCO 1M Benchmark
This executes the definitive real-world test. 
*Note: On the very first run, it will take ~30 minutes to build the HNSW graph natively in C++. It will then save a binary file called `custom_1M.index`. All future runs will instantly load this cached index.*
```bash
python benchmark_real.py
```

### Experiment 3: Visualizing the Results
After `benchmark_real.py` finishes, it will generate a `detailed_metrics.json` file. You can visualize the query-by-query breakdown by running:
```bash
python plot_metrics.py
```
This will generate `scatter_hybrid_vs_ada.png` and `histogram_recalls.png`.
