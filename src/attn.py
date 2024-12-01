import torch
import torch.nn as nn
import logging

class MultiHeadAttention_v2(nn.Module):
    """Implements a basic multi-head attention mechanism.
    """
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


class SelfAttention_v2(nn.Module):
    """Implements a basic self-attention mechanism.
    """
    def __init__(
        self, d_in, d_out, context_length, dropout: float, qkv_bias: bool = False
    ):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        num_batch, num_tokens, d_in = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # transposing along dimensions 1 and 2, so that batch size is kept as dimension 0
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))
        context_len = attention_scores.shape[0]
        attention_scores = attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_scores = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_scores = self.dropout(attention_scores)

        context_vec = torch.matmul(attention_scores, values)
        return context_vec
