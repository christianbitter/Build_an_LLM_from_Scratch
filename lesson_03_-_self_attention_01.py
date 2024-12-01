# this builds on the previous example.
# we use an embedded sentence and run it through a simple self attention mechanism

import logging
from src import GPTDatasetV1, create_data_loader
import torch

def main():
    logging.basicConfig(level=logging.DEBUG)
    # instead of loading the real data, we create our playground data and embedding
    sentence = ["Your", "journey", "starts", "one", "step"]
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                           [0.55, 0.87, 0.66],
                           [0.57, 0.85, 0.64],
                           [0.22, 0.58, 0.33], 
                           [0.77, 0.25, 0.1],
                           [0.05, 0.8, 0.55]])
    
    # now compute the self attention for some arbitrary query, e.g., the first token - journey
    logging.info(f"Input shape: {inputs.shape}")

    # now, instead of computing it for one query vector, we can compute it for all query vectors
    attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    for i, xi in enumerate(inputs):
        for j, xj in enumerate(inputs):
            attention_scores[i, j] = torch.dot(xi, xj)
    # now scale the attention scores via softmax, so that the scores can be interpreted as probabilities
    scaled_attention = torch.softmax(attention_scores, dim=-1)
    logging.info(f"Scaled attention scores: {scaled_attention}")

    # instead of doing the dot product with an unrolled loop ourself, we can use the pytorch function
    attention_scores = torch.matmul(inputs, inputs.T)
    scaled_attention = torch.softmax(attention_scores, dim=-1)
    logging.info(f"Scaled attention scores: {scaled_attention}")
    # these are the same

    # now let's build the context vectors for all tokens - scaling them by the attention scores
    # which gives us the weighted sum of all tokens - indicating the importance/ or direction of similarity
    # in vector space of each token
    context_vec = torch.matmul(scaled_attention, inputs)
    logging.info(f"Context vector: {context_vec}")


if __name__ == "__main__":
    main()