# this builds on the previous example and moves into multi-head attention
# to begin with, we will stack multiple self-attention layers
# mult-head attention is a mechanism to allow the model to focus on different parts of the input sequence
# this is done by having attention in parallel
# each attention is learnt with different weights

import logging
from src import GPTDatasetV1, create_data_loader, SelfAttention_v2
import torch
import torch.nn as nn


class MultiHeadAttention_v1(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads: int,
        context_length: int,
        dropout: float,
        kqv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttention_v2(
                    d_in=d_in,
                    d_out=d_out,
                    context_length=context_length,
                    dropout=dropout,
                    qkv_bias=kqv_bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


def main():
    torch.manual_seed(123)
    logging.basicConfig(level=logging.DEBUG)
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

    mha = MultiHeadAttention_v1(
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
