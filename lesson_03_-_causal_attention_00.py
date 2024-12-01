# this builds on the previous example.
# now we build self-attention with trainable weights
# and we turn this into a compact class
# now instead of using matrix multiplications we use torch linear layers

import logging
from src import GPTDatasetV1, create_data_loader
import torch
import torch.nn as nn


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias: bool = False):
        super().__init__()

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        attention_scores = torch.matmul(queries, keys.T)
        attention_scores = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = torch.matmul(attention_scores, values)
        return context_vec


def main():
    torch.manual_seed(123)

    logging.basicConfig(level=logging.DEBUG)
    # instead of loading the real data, we create our playground data and embedding
    sentence = ["Your", "journey", "starts", "one", "step"]
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.1],
            [0.05, 0.8, 0.55],
        ]
    )

    # now test the self attention module
    d_in, d_out = inputs.shape[1], 2
    sa = SelfAttention_v2(d_in=d_in, d_out=d_out)
    context_vec = sa(inputs)

    logging.info(f"Context vector: {context_vec}")


if __name__ == "__main__":
    main()
