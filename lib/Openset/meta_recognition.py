import libmr  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_weibull_model(mean_vector, correct_z, num_classes, tail_size):
    weibull_models = {}
    num_max_attempts = 10
    
    distances = 1 - cosine_similarity(correct_z, mean_vector.reshape(1, -1))
    distances = np.sort(distances.ravel())
    sz = int(len(distances) * tail_size)
    largest_distances = distances[-sz:]
    
    weibull_models = libmr.MR()
    is_valid = False
    attempts = 0
    while not is_valid and attempts < num_max_attempts:
        weibull_models.fit_high(largest_distances, sz)
        is_valid = weibull_models.is_valid
        attempts += 1
        
    if attempts >= num_max_attempts:
        print(f"Weibull fit not successful after {num_max_attempts} attempts")
        
    return weibull_models

def calculate_outlier_probability(z, mean_vector, weibull_models, num_classes):
    outlier_probabilities = []
    distance = 1 - cosine_similarity(z.reshape(1, -1), mean_vector.reshape(1, -1))
    outlier_probabilities.append(weibull_models.w_score(distance.item()))
    return np.array(outlier_probabilities)