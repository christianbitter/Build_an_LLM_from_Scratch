# this builds on the previous example.
# now we build self-attention with trainable weights
# and we turn this into a compact class

import logging
from src import GPTDatasetV1, create_data_loader
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_q = nn.Parameter(torch.randn(d_in, d_out))
        self.W_k = nn.Parameter(torch.randn(d_in, d_out))
        self.W_v = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        queries = torch.matmul(x, self.W_q)
        keys = torch.matmul(x, self.W_k)
        values = torch.matmul(x, self.W_v)

        attention_scores = torch.matmul(queries, keys.T)
        attention_scores = torch.softmax(attention_scores / keys.shape[-1]**.5, dim=-1)

        context_vec = torch.matmul(attention_scores, values)
        return context_vec

def main():
    torch.manual_seed(123)

    logging.basicConfig(level=logging.DEBUG)
    # instead of loading the real data, we create our playground data and embedding
    sentence = ["Your", "journey", "starts", "one", "step"]
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                           [0.55, 0.87, 0.66],
                           [0.57, 0.85, 0.64],
                           [0.22, 0.58, 0.33], 
                           [0.77, 0.25, 0.1],
                           [0.05, 0.8, 0.55]])
    
    # now test the self attention module
    d_in, d_out = inputs.shape[1], 2
    sa = SelfAttention_v1(d_in=d_in, d_out=d_out)
    context_vec = sa(inputs)

    logging.info(f"Context vector: {context_vec}")


if __name__ == "__main__":
    main()