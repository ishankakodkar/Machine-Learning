cmake_minimum_required(VERSION 3.10)

project(LinearRegression)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add executable
add_executable(linear_regression main.cpp)

# Link Eigen library
target_include_directories(linear_regression PRIVATE ${EIGEN3_INCLUDE_DIR})

# Add optimization flags for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

# Optional: Add compiler-specific optimization flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
    target_compile_options(linear_regression PRIVATE -march=native)
endif()

# Print a message
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
