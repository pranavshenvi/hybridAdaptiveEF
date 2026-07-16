# Benchmark Setup Guide

This guide walks you through the steps required to run the `benchmark_paper.py` script, which compares the fixed Ada-ef approach with the new single-pass Cluster-Aware dynamic EF architecture.

## 1. Prerequisites

Ensure you have a Python environment set up with the necessary dependencies:

```bash
pip install numpy scipy scikit-learn h5py
```

## 2. Compiling the Custom C++ HNSW Bindings

The benchmark relies on a heavily modified version of the `hnswlib` C++ library, located in the `chao_hybrid_ada_ef` directory. This custom version includes our implementations of:
- `searchKnnAdaptive`: Used for the Ada-ef benchmark.
- `searchKnnDynamic`: The single-pass dynamic search used by the Cluster-Aware architecture.

Before running the benchmark, you must compile and install these bindings.

**On Windows:**
Open PowerShell or your command prompt, navigate to the `chao_hybrid_ada_ef` folder, and install it in editable mode:
```powershell
cd chao_hybrid_ada_ef
pip install -e .
cd ..
```
*(Note: You will need a C++ compiler installed, such as MSVC from Visual Studio Build Tools, to successfully compile the extensions.)*

**On Linux/macOS:**
```bash
cd chao_hybrid_ada_ef
pip install -e .
cd ..
```

## 3. Data Preparation

The `benchmark_paper.py` script evaluates the architectures on the **MS MARCO** passage retrieval dataset (1M passages, 384-dimensional MiniLM embeddings). 

You must place the following data files in the root directory (alongside `benchmark_paper.py`):
- `msmarco-1M.hdf5`: The corpus embeddings.
- `msmarco_qemb_train.npz`: The training/calibration queries.
- `msmarco_qemb_validation.npz`: The test/validation queries.

## 4. Running the Benchmark

Once the bindings are compiled and the data is in place, you can run the benchmark. 

On the first run, the script will automatically build and save the HNSW index (`custom_1M.index`) to disk. This might take around 30 minutes. Subsequent runs will quickly load this index from disk.

**To run the benchmark on Windows:**
Because the benchmark outputs box-drawing characters for its tables, you must ensure your terminal is using UTF-8 encoding:
```powershell
$env:PYTHONIOENCODING="utf-8"
python benchmark_paper.py
```

**To run the benchmark on Linux/macOS:**
```bash
python benchmark_paper.py
```
