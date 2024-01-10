# Assume k: budget for buffer size
# Assume B_t: new batch for training
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# Define a simple feature extraction model
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.pretrained_model = pretrained_model
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(pretrained_model.children(self))[:-1])

    def forward(self, x):
        # Forward pass to obtain features
        return self.features(x)

def agglomerative_cluster(B_t, M, k):
    ''' 
    Key arguments: New Batch(B_t), Buffer, budget (k)
    here B_t have n data points in the format (x, y)
    
    '''
    for (x_i, y_i) in B_t:
        # Create an instance of the FeatureExtractor
        feature_extractor = FeatureExtractor(nn.Module)
        # Extract features
        a_i = feature_extractor(x_i)
        if len(M) < k:
            M.append((x_i, y_i, a_i, 1))
        
        else:
            j = min(range(len(M)), key=lambda j: np.linalg.norm(a_i - M[j][2]))

            a_prime = (M[j][3] * M[j][2] + a_i) / (1 + M[j][3])

            if np.linalg.norm(a_i - a_prime) <= np.linalg.norm(M[j][2] - a_prime):
                M[j] = (x_i, y_i, a_prime, M[j][3])   
    return M

# Example usage:
pretrained_model = models.resnet50(weights=True)
B_t = [(torch.randn((3, 224, 224)), 0) for _ in range(10)]  # Replace this with your actual data
M = []
k = 10
M = agglomerative_cluster(B_t, M, k)
print(len(M))