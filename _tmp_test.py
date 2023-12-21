import torch
import torch.nn as nn


# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)

# Activate module
layer_norm(embedding)

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)

# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])

layer_norm2 = nn.LayerNorm([C, H, W])

output = layer_norm(input)
layer_1_result = output.clone()

output = layer_norm2(output)
layer_2_result = output.clone()

print(layer_1_result - layer_2_result)