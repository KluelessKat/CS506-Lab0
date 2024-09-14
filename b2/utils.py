import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the scalar dot product of the two vectors.
    # Hint: use `np.dot`.
    '''
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    
    # Note: The cosine similarity is a commonly used similarity 
    metric between two vectors. It is the cosine of the angle between 
    two vectors, and always between -1 and 1.
    
    # The formula for cosine similarity is: 
    # (v1 dot v2) / (||v1|| * ||v2||)
    
    # ||v1|| is the 2-norm (Euclidean length) of the vector v1.
    
    # Hint: Use `dot_product` and `np.linalg.norm`.
    '''
    norm_v1 = np.linalg.norm(v1)  # Euclidean norm of v1
    norm_v2 = np.linalg.norm(v2)  # Euclidean norm of v2
    dot_prod = dot_product(v1, v2)  # Dot product using the previously defined function
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Handle edge case where one of the vectors has zero magnitude
    
    return dot_prod / (norm_v1 * norm_v2)  # Cosine similarity formula

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    Return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    
    # Hint: You should use the cosine_similarity function that you already wrote.
    # Hint: For this lab, you can just use a for loop to iterate through vectors.
    '''
    best_similarity = -1  # Start with the lowest possible cosine similarity
    best_index = -1  # Initialize to -1 to signify no valid index yet

    #for i, vec in enumerate(vectors):
    #     sim = cosine_similarity(target_vector, vec)  # Compute cosine similarity
    #     if sim > best_similarity:  # Check if this is the highest similarity found
    #         best_similarity = sim
    #         best_index = i  # Update index with the current best match
    
    # return best_index  # Return the index of the most similar vector
    similarities = [cosine_similarity(target_vector, vec) for vec in vectors]
    return np.argmax(similarities)
