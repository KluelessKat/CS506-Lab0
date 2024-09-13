## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    # (v1 dot v2) / (||v1|| * ||v2||)
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = dot_product(vector1, vector2)/((np.linalg.norm(vector1))*np.linalg.norm(vector2))
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    ### YOUR CODE HERE
    vectors = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    target_vector = np.array([2, 3, 4])
    
    # Call the nearest_neighbor function
    result = nearest_neighbor(target_vector, vectors)
    
    # Expected nearest neighbor is the first vector because it's closest to [2, 3, 4]
    expected_index = 0  # Manually computed (smallest Euclidean distance)
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
