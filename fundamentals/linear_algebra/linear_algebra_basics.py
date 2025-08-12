import numpy as np

# --- Vectors ---
print("--- Vectors ---")

# Create a vector (1D array)
v = np.array([1, 2, 3])
print(f"Vector v: {v}")

# Create another vector
u = np.array([4, 5, 6])
print(f"Vector u: {u}")

# Vector addition
print(f"v + u = {v + u}")

# Vector subtraction
print(f"v - u = {v - u}")

# Scalar multiplication
print(f"2 * v = {2 * v}")

# Dot product
print(f"v . u = {np.dot(v, u)}")
print(f"v . u = {v @ u}") # Another way to calculate dot product

# Vector norm (magnitude)
print(f"Magnitude of v: {np.linalg.norm(v)}")


# --- Matrices ---
print("\n--- Matrices ---")

# Create a matrix (2D array)
A = np.array([[1, 2], [3, 4]])
print(f"Matrix A:\n{A}")

# Create another matrix
B = np.array([[5, 6], [7, 8]])
print(f"Matrix B:\n{B}")

# Matrix addition
print(f"A + B:\n{A + B}")

# Matrix multiplication (element-wise)
print(f"A * B (element-wise):\n{A * B}")

# Matrix multiplication (dot product)
print(f"A @ B (dot product):\n{np.dot(A, B)}")
print(f"A @ B (dot product):\n{A @ B}")

# Transpose of a matrix
print(f"Transpose of A:\n{A.T}")

# Determinant of a matrix
print(f"Determinant of A: {np.linalg.det(A)}")

# Inverse of a matrix
try:
    A_inv = np.linalg.inv(A)
    print(f"Inverse of A:\n{A_inv}")
    # Verify the inverse
    print(f"A @ A_inv (should be identity matrix):\n{A @ A_inv}")
except np.linalg.LinAlgError as e:
    print(f"Could not compute inverse of A: {e}")


# --- Eigenvalues and Eigenvectors ---
print("\n--- Eigenvalues and Eigenvectors ---")
C = np.array([[2, -1], [4, -3]])
print(f"Matrix C:\n{C}")

eigenvalues, eigenvectors = np.linalg.eig(C)
print(f"Eigenvalues of C: {eigenvalues}")
print(f"Eigenvectors of C:\n{eigenvectors}")

# Verify: C * v = lambda * v for each eigenvector v and eigenvalue lambda
for i in range(len(eigenvalues)):
    lambda_val = eigenvalues[i]
    v_vec = eigenvectors[:, i]
    print(f"C @ v_{i+1}: {C @ v_vec}")
    print(f"lambda_{i+1} * v_{i+1}: {lambda_val * v_vec}")
    # np.testing.assert_allclose is used for comparing floating point numbers
    np.testing.assert_allclose(C @ v_vec, lambda_val * v_vec)
    print(f"Verification for eigenvalue {lambda_val:.2f} and its eigenvector passed.")
