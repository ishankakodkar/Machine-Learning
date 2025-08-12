# C++ Implementation of Linear Regression

This directory contains a C++ implementation of simple linear regression. To keep it fast and low-latency, it uses the **Normal Equation** method to solve for the optimal model parameters directly, avoiding iterative gradient descent. This is very efficient for inference, especially with a moderate number of features.

## Dependencies

This project relies on the **Eigen** library, a high-performance C++ template library for linear algebra.

### Installing Eigen

You need to download Eigen and make it available to the compiler. You can download it from the official website or use a package manager.

**Option 1: Download**

1.  Download the latest version of Eigen from [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/).
2.  Unzip the downloaded file. This will create a directory (e.g., `eigen-3.4.0/`). Eigen is a header-only library, so there's nothing to compile.

**Option 2: Homebrew (macOS)**

```bash
brew install eigen
```

## Building the Project

This project uses CMake to manage the build process. 

1.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

2.  **Run CMake:**

    If you downloaded Eigen manually, you need to tell CMake where to find it. Replace `/path/to/your/eigen/directory` with the actual path.

    ```bash
    cmake .. -DCMAKE_PREFIX_PATH=/path/to/your/eigen/directory
    ```

    If you installed Eigen with Homebrew, CMake should find it automatically:

    ```bash
    cmake ..
    ```

3.  **Compile the code:**

    ```bash
    make
    ```

4.  **Run the executable:**

    ```bash
    ./linear_regression
    ```
