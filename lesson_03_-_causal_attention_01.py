# this builds on the previous example.
# now we build self-attention with trainable weights
# and we turn this into a compact class
# now instead of using matrix multiplications we use torch linear layers
# simply extend to batch size 2

import logging
from src import GPTDatasetV1, create_data_loader, SelfAttention_v2
import torch
import torch.nn as nn


def main():
    torch.manual_seed(123)

    logging.basicConfig(level=logging.DEBUG)
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
    # now test the self attention module
    d_in, d_out = inputs.shape[2], 2

    logging.info("SelfAttention_v2 with batch size 2")
    sa = SelfAttention_v2(
        d_in=d_in, d_out=d_out, context_length=inputs.shape[1], dropout=0.0
    )
    context_vec = sa(inputs)

    logging.info(f"Context vector: {context_vec}")


if __name__ == "__main__":
    main()
