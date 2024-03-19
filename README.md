# KNN_CUDA

## Description

This project provides an implementation of the k-Nearest Neighbors (KNN) algorithm utilizing CUDA for accelerated parallel processing. It aims to exploit NVIDIA GPU computing power to speed up the distance calculation process inherent in the KNN algorithm, which is widely used in machine learning for classification and regression tasks. This CUDA-based approach is designed to offer significant performance improvements, especially for high-dimensional data.

## Features

- **CUDA Acceleration**: Leverages CUDA to parallelize the distance calculation process, enabling rapid processing of large datasets.
- **Efficient Data Handling**: Utilizes the Thrust library for efficient data sorting and manipulation on the GPU.
- **Scalability**: Designed to efficiently handle large volumes of data with high dimensionality.
- **Versatility**: Applicable to a wide range of classification and regression tasks in machine learning and data science.

## Prerequisites

Before running this project, ensure you have the following:

- NVIDIA GPU with CUDA Compute Capability 3.5 or higher.
- CUDA Toolkit 10.0 or newer installed on your system.
- C++ compiler compatible with the installed version of CUDA Toolkit.

## Installation

To get started with the KNN_CUDA project, clone the repository to your local machine:

```bash
git clone https://github.com/kazi-ishrak/KNN_CUDA.git
cd KNN_CUDA

## Usage

Follow these steps to use the KNN_CUDA project:

1. Prepare your dataset following the format specified in the `final_data.txt` and `test_data.txt` files.
2. Compile the project using the following command:

    ```bash
    nvcc -o knn_cuda main.cu
    ```

3. Execute the compiled binary:

    ```bash
    ./knn_cuda
    ```

You will be prompted to enter the number of reference points, the number of test points, and the value of `k`.


