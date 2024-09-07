import numpy as np

def skew(vector: np.ndarray) -> np.ndarray:
    """transfer to Skew-Symmetric Matrix

    Args:
        vector (ndarray): vector with shape(3, 1)

    Returns:
        ndarray: Skew-Symmetric Matrix with shape (3, 3)
    """
    matrix = np.zeros((3, 3))

    matrix[0, 1] = -vector[2]
    matrix[0, 2] = vector[1]

    matrix[1, 0] = vector[2]
    matrix[1, 2] = -vector[0]

    matrix[2, 0] = -vector[1]
    matrix[2, 1] = vector[0]

    return matrix
