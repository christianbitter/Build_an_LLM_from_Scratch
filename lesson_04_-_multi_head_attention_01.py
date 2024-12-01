# this builds on the previous example and moves into multi-head attention
# to begin with, we will stack multiple self-attention layers
# mult-head attention is a mechanism to allow the model to focus on different parts of the input sequence
# this is done by having attention in parallel
# each attention is learnt with different weights
# now we will implement a real model with multi-head attention

import logging
from src import GPTDatasetV1, create_data_loader, SelfAttention_v2
import torch
import torch.nn as nn


class MultiHeadAttention_v2(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads:int, context_length:int, dropout: float, kqv_bias: bool = False
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by no_heads"

        self.no_heads = num_heads
        self.d_out = d_out
        self.d_in = d_in
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=kqv_bias)
        
        # an additional linear projection to project the concatenated heads
        self.out_projection = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        no_batch, no_tokens, d_in = x.shape

        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        # project the keys, queries and values into no_heads
        # this effectively splits up the single head k-q-v into multiple heads no-head k-q-v 
        keys = keys.view(no_batch, no_tokens, self.no_heads, self.head_dim)
        queries = queries.view(no_batch, no_tokens, self.no_heads, self.head_dim)
        values = values.view(no_batch, no_tokens, self.no_heads, self.head_dim)

        # transpose from (no_batch, no_tokens, no_heads, head_dim) to (no_batch, no_heads, no_tokens, head_dim)
        # this brings the data back to have batches per attention head
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # calculate the attention scores - we could also transpose in the last two axis as opposed to 2,3
        attention_scores = torch.matmul(queries, keys.transpose(2, 3))

        mask_bool = self.mask.bool()[:no_tokens, :no_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)
        attention_scores = torch.softmax(
            attention_scores / self.head_dim ** 0.5, dim=-1
        )
        attention_scores = self.dropout(attention_scores)

        # now multiply the attention scores with the values
        context_vec = torch.matmul(attention_scores, values)

        # we place the output of the individual heads now contiguously, so that we concatenate them
        context_vec = context_vec.transpose(1, 2).contiguous().view(no_batch, no_tokens, self.d_out)
        context_vec = self.out_projection(context_vec)

        return context_vec


def main():
    torch.manual_seed(123)
    logging.basicConfig(level=logging.DEBUG)
    logging.info(__name__)
    logging.info(f"torch version: {torch.__version__}")
    logging.info(f"torch cuda available: {torch.cuda.is_available()}")
    # instead of loading the real data, we create our playground data and embedding
    sentence = ["Your", "journey", "starts", "one", "step"]
    inputs = torch.tensor(
        [
            [
                [0.43, 0.15, 0.89],
                [0.55, 0.87, 0.66],
                [0.57, 0.85, 0.64],
                [0.22, 0.58, 0.33],
                [0.77, 0.25, 0.1],
                [0.05, 0.8, 0.55],
            ]
        ]
    )
    logging.info(f"batch: {inputs.shape}")

    # play with the output dimension to see how it changes
    d_in, d_out = inputs.shape[2], 2
    logging.info(f"{d_in=}, {d_out=}")

    mha = MultiHeadAttention_v2(
        d_in=d_in,
        d_out=d_out,
        num_heads=2,
        context_length=6,
        dropout=0.1,
        kqv_bias=False,
    )

    outs = mha(inputs)
    logging.info(f"{outs.shape=}")
    logging.info(f"{outs=}")


if __name__ == "__main__":
    main()
