# DSA5208-Parallelizing-Algorithm-with-MPIX
This repository contains the codes and documents for DSA5208 Assignment 3 - Parallelizing Algorithm with MPIX

### Software Used
![C/C++](https://img.shields.io/badge/C/C++-Programming-blue.svg?style=flat&logo=c&logoColor=FFF)
![Open MPI](https://img.shields.io/badge/Open_MPI-Distributed-1f4e79.svg?style=flat&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTMgMjBWNGgzLjJMMTIgMTJsNS44LThIMjF2MTZoLTNWOS41bC02IDgtNi04VjIweiIvPjwvc3ZnPg%3D%3D)
![OpenMP](https://img.shields.io/badge/OpenMP-Multithreading-5c7cfa.svg?style=flat&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTMgMjBWNGgzLjJMMTIgMTJsNS44LThIMjF2MTZoLTNWOS41bC02IDgtNi04VjIweiIvPjwvc3ZnPg%3D%3D)
![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-GPU_Offload-76B900.svg?style=flat&logo=nvidia&logoColor=76B900)
![Bash](https://img.shields.io/badge/Bash-Shell-dd4814?style=flat&&logo=gnu-bash&logoColor=FFF)


### Team Members
|  Matriculation Number | Name                |
| ---                   | ---                 |
| A0220825M             | Koh Si Xing         |
| A0221663L             | Ashley Lee Ruoxin   |
| A0333470L             | Kevin Royce Thomson | 


### Main Files/Folders
| File / Folder                     | Description                                                                                              |
| ---                               | ---                                                                                                      |
| als_serial.cpp                    | Serial ALS baseline for correctness, speedup and serial scaling.                                         |
| als_mpi.cpp                       | MPI ALS with user and movie blocks distributed across processes and factors shared via `MPI_Allgatherv`. |
| als_hybrid.cpp                    | MPI + OpenMP ALS, using threads to parallelise updates within each process.                              |
| als_gpu.cpp                       | MPI + GPU ALS, using OpenMP GPU offloading for user and movie updates.                                   |
| all_results_long.csv              | Results by run and iteration, including timings, RMSE, resources and dataset size.                       |
| Data Cleaning and Analysis.ipynb  | Prepares dataset subsets and produces the report's charts and tables.                                    |
| DSA5208 Assignment 3 Report.pdf   | Report for this Assignment                                                                               |

### Dataset
MovieLens 32M (`ratings.csv`), from [GroupLens](https://grouplens.org/datasets/movielens/32m/):
```bash
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip && unzip ml-32m.zip
cp ml-32m/ratings.csv ./ratings.csv
```
The weak-scaling subsets (`ratings_1of64.csv`, `ratings_1of16.csv`, `ratings_1of4.csv`) are made in `data cleaning and analysis.ipynb` by picking random users (fixed seed 42) with all of their ratings.

### Requirements
- Linux with a C++17 compiler (`g++`) and Open MPI
- GPU version only: an NVIDIA GPU
- Notebook (OPTIONAL; For Graphs and Splitting Dataset): Python 3 with `pandas` and `matplotlib`

### How to Compile
```bash
# Serial
g++ -O2 -std=c++17 -o als_serial als_serial.cpp

# MPI
mpicxx -O2 -std=c++17 -o als_mpi als_mpi.cpp

# MPI + OpenMP
mpicxx -O2 -std=c++17 -fopenmp -o als_hybrid als_hybrid.cpp

# MPI + GPU (with the NVIDIA HPC SDK compilers and MPI on the PATH)
mpicxx -O2 -std=c++17 -mp=gpu -gpu=ccnative -Minfo=mp -o als_gpu als_gpu.cpp
```

### How to Run
With one data file, 20% of the ratings are held out at random (fixed seed) as the test set.
```bash
# Serial
./als_serial ratings.csv

# MPI, e.g. 64 processes
mpirun -np 64 --bind-to core ./als_mpi ratings.csv

# MPI + OpenMP, e.g. 64 processes x 4 threads = 256 cores
mpirun -np 64 --map-by slot:PE=4 -x OMP_NUM_THREADS=4 ./als_hybrid ratings.csv

# MPI + GPU, 1 process on 1 GPU 
mpirun -np 1 -x OMP_TARGET_OFFLOAD=MANDATORY -x CUDA_VISIBLE_DEVICES=0 ./als_gpu ratings.csv
```