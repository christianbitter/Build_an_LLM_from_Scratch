# this builds on the previous example.
# now we build self-attention with trainable weights

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
    
    # for illustrative purposes we fix the input to the second token
    x_2 = inputs[1]


    # let's define the trainable weight matrices - corresponding to
    # the query vector, which was fixed to the second input
    # the key vector, which is all the elements in our input
    # the value vector, which is what we want to compute
    d_in, d_out = inputs.shape[1], 2

    W_q = torch.nn.Parameter(torch.randn(d_in, d_out))
    W_k = torch.nn.Parameter(torch.randn(d_in, d_out))
    W_v = torch.nn.Parameter(torch.randn(d_in, d_out))

    # for illustrative purposes, we compute everything in the context of the same token
    q_2 = torch.matmul(x_2, W_q)
    k_2 = torch.matmul(x_2, W_k)
    v_2 = torch.matmul(x_2, W_v)

    logging.info("Computing self attention for the second token")
    logging.info(f"Query 2: {q_2}")
    logging.info(f"Shapes: {W_q.shape}, {W_k.shape}, {W_v.shape}")
    logging.info(f"Shapes: {q_2.shape}, {k_2.shape}, {v_2.shape}")

    # now let's do this across all the tokens
    logging.info("Computing across all tokens")
    keys = torch.matmul(inputs, W_k)
    queries = torch.matmul(inputs, W_q)
    values = torch.matmul(inputs, W_v)
    logging.info(f"Shapes: {queries.shape}, {keys.shape}, {values.shape}")

    # now let's replicate our scaled self-attention computation for the second token again
    keys_2 = keys[2] # pull out the keys for the second token
    attention_scores_22 = q_2.dot(keys_2)
    logging.info(f"Attention scores 22: {attention_scores_22}")

    # generalize to all elements
    attention_scores_22 = torch.matmul(q_2, keys.T)
    logging.info(f"Attention scores 22: {attention_scores_22}")

    # apply scaled dot product attention by scaling by the number of dimensions in the key
    d_k = keys.shape[-1]
    attention_scores = torch.softmax(attention_scores_22 / d_k**.5, dim=-1)
    logging.info(f"Scaled attention scores: {attention_scores}")    

    # finally, compute the context vector as a weighted sum of the values
    context_vec = torch.matmul(attention_scores, values)
    logging.info(f"Context vector: {context_vec}")


if __name__ == "__main__":
    main()