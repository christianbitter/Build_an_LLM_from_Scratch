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
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)

    logging.info(f"Self attention scores: {attn_scores_2}")
    # now scale the attention scores via softmax, so that the scores can be interpreted as probabilities
    scaled_attention_2 = torch.softmax(attn_scores_2, dim=0)
    logging.info(f"Scaled attention scores: {scaled_attention_2}")

    # finally each embedded token is scaled by the attention score and summed up
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        logging.info(f"Token: {x_i} --> Attention: {scaled_attention_2[i]}")
        context_vec_2 += scaled_attention_2[i] * x_i
    
    logging.info(f"Context vector: {context_vec_2}")


if __name__ == "__main__":
    main()