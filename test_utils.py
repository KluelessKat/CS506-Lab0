import numpy as np
import pytest
from utils import dot_product, cosine_similarity, nearest_neighbor

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    # Example vectors
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    # Call the cosine_similarity function
    result = cosine_similarity(vector1, vector2)
    
    # Calculate expected cosine similarity manually
    dot_product_val = np.dot(vector1, vector2)  # = 32
    norm1 = np.linalg.norm(vector1)             # = sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    norm2 = np.linalg.norm(vector2)             # = sqrt(4^2 + 5^2 + 6^2) = sqrt(77)
    expected_result = dot_product_val / (norm1 * norm2)  # 32 / (sqrt(14) * sqrt(77))
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    # Define a set of vectors (rows) and a target vector
    target_vector = np.array([1, 2])
    vectors = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    
    # Call the nearest_neighbor function
    result = nearest_neighbor(target_vector, vectors)
    
    # Expected nearest neighbor is the first vector because it's closest to [2, 3, 4]
    expected_index = 0  # Manually computed (smallest Euclidean distance)
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
