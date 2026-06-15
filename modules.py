from einops import rearrange, einsum
import math, torch
import torch.nn as nn
from typing import Optional, Callable
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np



from collections import defaultdict


import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device=None, dtype: torch.dtype=None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(in_features, 
                                               out_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std, 
                                    a=-3 * std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W

from torch import Tensor
from jaxtyping import Float
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    l = Linear(d_in, d_out)
    l.W.data = weights.T
    return l(in_features)


import math, torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim,
                                               device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_weights, mean=0, std=1,
                                    a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_weights[token_ids]

import math, torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5,
                 device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _calc_rms(self, a):
        return torch.sqrt(((a.pow(2)).sum(dim=-1, keepdim=True)) / self.d_model + self.eps)
    def _calc_rmsnorm(self, rms, a, g):
        return a * g / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(): (batch, seq_len, d_model)
        
        in_dtype = x.dtype
        x = x.to(torch.float32)


        rms = self._calc_rms(x)
        # rms.size(): (batch, seq_len, 1)
        result = self._calc_rmsnorm(rms, x, self.g)

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, w1_w, w2_w, w3_w):
        # d_ff 应满足约为 8/3 * d_model 且是 64 的倍数
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.w1.W.data = w1_w.T
        self.w2.W.data = w2_w.T
        self.w3.W.data = w3_w.T
        # Linear 实现：存储 W 的转置(用的是 x @ W 而非 W @ x )

    def _SiLU(self, x):
        return x * torch.sigmoid(x)
    def _SwiGLU(self, x):
        return self.w2(self._SiLU(self.w1(x)) * self.w3(x))

    def forward(self, x):
        return self._SwiGLU(x)


# RoPE: inject positional infos. Rotary Position Embeddings
# 修改 K, V 对每一组二维坐标分别应用，每一组使用的旋转基数 θ 是不同的
from jaxtyping import Float, Bool, Int
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                 device: torch.device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, freq)

        R = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer("cos", R.cos(), persistent=False)
        self.register_buffer("sin", R.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_i = self.cos[token_positions].to(x.dtype)
        sin_i = self.sin[token_positions].to(x.dtype)
        while cos_i.ndim < x.ndim:
            cos_i = cos_i.unsqueeze(-3)
            sin_i = sin_i.unsqueeze(-3)
        # Avoid expensive matmuls
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = -x[..., 1::2]
        x_rotated[..., 1::2] =  x[..., 0::2]
        return (x * cos_i) + (x_rotated * sin_i)

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    r = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return r(in_query_or_key, token_positions)


# Scaled Dot-Product Attention

# Softmax
from jaxtyping import Float, Bool

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    v_max = torch.max(in_features, dim=dim, keepdim=True)[0]
    in_features = in_features - v_max
    exp_sum = torch.exp(in_features).sum(dim=dim, keepdim=True)
    return torch.exp(in_features) / exp_sum


# Attention

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k_scale = 1.0 / math.sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        QKT = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
        QKT = QKT * self.d_k_scale

        if mask is not None:
            QKT = QKT.masked_fill(~mask, float("-inf"))
        
        QKT = run_softmax(QKT, dim=-1)
        return einsum(QKT, V, "... n m, ... m d_k -> ... n d_k")

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"]  = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"]  ): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    sdpa = ScaledDotProductAttention(Q.size(-1))
    return sdpa(Q, K, V, mask=mask)
    



# class MultiheadSelfAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int,
#                  max_seq_len: int, rope_theta: float = 10000.0,
#                  use_rope: bool=True, device=None, dtype=None):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.use_rope = use_rope
#         self.attn = ScaledDotProductAttention(self.d_k)
#         mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
#         self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

#         if use_rope:
#             self.RoPE = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k,
#                                                   max_seq_len=max_seq_len, device=device)

#     def forward(self, x, token_positions=None):
#         q = self.q_proj(x)
#         q = rearrange(q, "batch seq_len (head_num d_k) -> batch head_num seq_len d_k",
#                       head_num=self.num_heads)
#         k = self.k_proj(x)
#         k = rearrange(k, "batch seq_len (head_num d_k) -> batch head_num seq_len d_k",
#                       head_num=self.num_heads)
#         v = self.v_proj(x)
#         v = rearrange(v, "batch seq_len (head_num d_k) -> batch head_num seq_len d_k",
#                       head_num=self.num_heads)

#         if self.use_rope:
#             q,k = self.RoPE(q, token_positions), self.RoPE(k, token_positions)
#         seq_len = x.size(1)
#         out = self.attn(q, k, v, mask=self.causal_mask[..., :seq_len, :seq_len])
#         out = rearrange(out, "... head_num seq_len d_k -> ... seq_len (head_num d_k)")

#         return self.o_proj(out)


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        # QKV projection (fused-friendly layout)
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE (assumes you already implemented efficient version)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(self, x, token_positions=None):
        """
        x: (B, S, D)
        """

        B, S, _ = x.shape
        H = self.num_heads

        # QKV projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape -> (B, H, S, D/H)
        q = q.view(B, S, H, self.d_head).transpose(1, 2)
        k = k.view(B, S, H, self.d_head).transpose(1, 2)
        v = v.view(B, S, H, self.d_head).transpose(1, 2)

        # RoPE (vectorized, no Python loops)
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 直接替代所有 mask + softmax + matmul
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
        )

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        return self.o_proj(out)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-h
def run_multihead_self_attention(
    d_model: int,
    num_heaead attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    max_seq_len = in_features.shape[-2]
    cmhsa = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=False,
                                  max_seq_len=max_seq_len, device=in_features.device, dtype=in_features.dtype)
    cmhsa.load_state_dict({
        "q_proj.W": q_proj_weight.T,
        "k_proj.W": k_proj_weight.T,
        "v_proj.W": v_proj_weight.T,
        "o_proj.W": o_proj_weight.T,
    })
    return cmhsa(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"]  = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"]  ): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    cmhsa = MultiheadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        rope_theta=theta,
        use_rope=(token_positions is not None),
        max_seq_len=max_seq_len,
        device=in_features.device,
        dtype=in_features.dtype
    )
    cmhsa.load_state_dict({
        "q_proj.W": q_proj_weight.T,
        "k_proj.W": k_proj_weight.T,
        "v_proj.W": v_proj_weight.T,
        "o_proj.W": o_proj_weight.T,
    })

    return cmhsa(in_features,token_positions)




def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    eps = 1e-5
    batch_size, seq_len, _ = in_features.shape
    device = in_features.device
    dtype = in_features.dtype
    token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    rmsnorm1 = run_rmsnorm(d_model, eps, weights['ln1.weight'], in_features)
    multihead = run_multihead_self_attention_with_rope(d_model, num_heads,
                                                       max_seq_len, theta,
                                                       weights['attn.q_proj.weight'],
                                                       weights['attn.k_proj.weight'],
                                                       weights['attn.v_proj.weight'],
                                                       weights['attn.output_proj.weight'],
                                                       rmsnorm1, token_positions)
    multihead = multihead + in_features
    rmsnorm2 = run_rmsnorm(d_model, eps, weights['ln2.weight'], multihead)
    ff = run_swiglu(d_model, d_ff, weights['ffn.w1.weight'], weights['ffn.w2.weight'], 
                    weights['ffn.w3.weight'], rmsnorm2)
    output = ff + multihead
    return output

from collections import defaultdict

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    eps = 1e-5
    
    embeddings = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], 
                              in_indices)
    
    transformer_num_layer = embeddings
    for num_layer in range(num_layers):
        weights_num_layer = defaultdict(Tensor)
        weights_num_layer['attn.q_proj.weight'] = weights[f'layers.{num_layer}.attn.q_proj.weight']
        weights_num_layer['attn.k_proj.weight'] = weights[f'layers.{num_layer}.attn.k_proj.weight']
        weights_num_layer['attn.v_proj.weight'] = weights[f'layers.{num_layer}.attn.v_proj.weight']
        weights_num_layer['attn.output_proj.weight'] = weights[f'layers.{num_layer}.attn.output_proj.weight']
        weights_num_layer['ln1.weight'] = weights[f'layers.{num_layer}.ln1.weight']
        weights_num_layer['ln2.weight'] = weights[f'layers.{num_layer}.ln2.weight']
        weights_num_layer['ffn.w1.weight'] = weights[f'layers.{num_layer}.ffn.w1.weight']
        weights_num_layer['ffn.w2.weight'] = weights[f'layers.{num_layer}.ffn.w2.weight']
        weights_num_layer['ffn.w3.weight'] = weights[f'layers.{num_layer}.ffn.w3.weight']
        
        transformer_num_layer = run_transformer_block(d_model, num_heads, 
                                                     d_ff, context_length, 
                                                     rope_theta, weights_num_layer, 
                                                     transformer_num_layer)
        
    norm = run_rmsnorm(d_model, eps, weights['ln_final.weight'], transformer_num_layer)
    output = run_linear(d_model, vocab_size, weights['lm_head.weight'], norm)
    return output



# 4 Training a Transformer LM

# 4.1 Cross-entropy loss

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs_max = torch.max(inputs, dim=-1, keepdim=True)[0]
    inputs = inputs - inputs_max
    inputs_exp_sum = torch.exp(inputs).sum(dim=-1, keepdim=True)
    o_softmax_log = inputs - torch.log(inputs_exp_sum)
    return -o_softmax_log.gather(1, targets.unsqueeze(1)).squeeze(1).mean()


import torch

def run_cross_entropy_for_gem(logits: torch.Tensor, labels: torch.Tensor, ignore_index=-666):
    B, S, V = logits.shape
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    logits = logits.reshape(-1, V)
    labels = labels.reshape(-1)

    mask = labels != ignore_index
    logits = logits[mask]
    labels = labels[mask]

    loss = run_cross_entropy(logits, labels)
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta_1 = betas[0]
        beta_2 = betas[1]
        lamb = weight_decay
        defaults = {"lr": lr, "beta_1": beta_1, "beta_2": beta_2, "lamb": lamb, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            lamb = group["lamb"]

            for p in group["params"]:
                if p.grad is None:
                    continue


                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                t = state.get("t", 0) + 1 # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                lr_t = lr * (math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t))
                p.data -= lr * lamb * p.data
                state["m"].mul_(beta_1).add_(grad, alpha=1 - beta_1)
                state["v"].mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps) # Update weight tensor in-place.
                state["t"] = t  # Increment iteration number. 

        return loss


def get_adamw_cls() :
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate + ((1 + 
                                     math.cos(
                                         math.pi * (
                                             (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
                                    ) * 
                                    (max_learning_rate - min_learning_rate)) / 2
    else:
        return min_learning_rate


# 5 Training loop

# A data loader turns this into a stream of batches, where each batch consists of 𝐵 sequences of length 𝑚

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm().item() ** 2
    g_l2 = math.sqrt(total_norm_sq)
    if g_l2 < max_l2_norm:
        return None
    else:
        scale_c = max_l2_norm / (g_l2 + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale_c)
        return None


# 支持 数据集用 memmap 优化

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = dataset.shape[-1] - context_length

    rand_is = torch.randint(0, max_start, (batch_size,))
    # 生成一个包含 batch_size 个随机整数的张量，每个整数的取值范围是 [0, max_start)（即 0 到 max_start-1 之间的整数）。

    ret = (
        torch.zeros(batch_size, context_length, dtype=torch.long, device=device),
        torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    )

    for idx in range(batch_size):
        ix = torch.randint(0, max_start, (batch_size,))

        x = torch.stack([
            torch.from_numpy(dataset[i:i+context_length].astype(np.int64))
            for i in ix
        ])

        y = torch.stack([
            torch.from_numpy(dataset[i+1:i+context_length+1].astype(np.int64))
            for i in ix
        ])

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)


    return ret

from typing import BinaryIO, IO
import os

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str ,
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {}
    obj["model"] = model.state_dict()
    obj["optim"] = optimizer.state_dict()
    obj["it"] = iteration
    torch.save(obj, out)


import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def run_load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    obj = torch.load(src, map_location="cpu")
    state_dict = obj["model"]

    if isinstance(model, DDP):
        state_dict = {
            f"module.{k}": v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(obj["optim"])

    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    return obj["it"]

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device  = None,
        dtype: torch.dtype  = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device  = None,
        dtype: torch.dtype  = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " batch sequence_length"]  = None,
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        batch_size, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=x.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Pre-norm attention block
        h = x + self.attn(self.ln1(x), token_positions)

        # Pre-norm FFN block
        out = h + self.ffn(self.ln2(h))

        return out

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device  = None,
        dtype: torch.dtype  = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

        batch_size, seq_len = in_indices.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds context length {self.context_length}"
            )

        x = self.token_embeddings(in_indices)

        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
        ).unsqueeze(0).expand(batch_size, -1)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", valid_data)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = run_get_batch(
                dataset=data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            loss = run_cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


from torch.distributions import Categorical

def top_p_sampling(probs, top_p=0.9, eps=1e-12):
    probs = probs.clone()
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1,
    )
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(
        sorted_indices_to_remove,
        0.0,
    )
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs,
    )
    filtered_probs = filtered_probs / filtered_probs.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    return filtered_probs


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    if penalty is None or penalty <= 1.0 or len(generated_ids) == 0:
        return logits

    logits = logits.clone()
    for token_id in set(generated_ids):
        token_logit = logits[token_id]
        if token_logit < 0:
            logits[token_id] = token_logit * penalty
        else:
            logits[token_id] = token_logit / penalty
    return logits


def apply_no_repeat_ngram_mask(
    logits: torch.Tensor,
    generated_ids: list[int],
    ngram_size: int = 4,
) -> torch.Tensor:
    if ngram_size is None or ngram_size <= 1:
        return logits

    if len(generated_ids) < ngram_size - 1:
        return logits

    prefix_to_next = {}
    for i in range(len(generated_ids) - ngram_size + 1):
        prefix = tuple(generated_ids[i : i + ngram_size - 1])
        next_token = generated_ids[i + ngram_size - 1]
        prefix_to_next.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = prefix_to_next.get(current_prefix)
    if not banned:
        return logits

    logits = logits.clone()
    for token_id in banned:
        logits[token_id] = float("-inf")
    return logits


def generating(
    model,
    enc_user_prompt,
    end_token: int,
    context_len: int = 256,
    max_token: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int | None = None,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
    ban_token_ids: list[int] | None = None,
):
    model.eval()

    device = next(model.parameters()).device

    enc_user_prompt = torch.tensor(
        [enc_user_prompt],
        dtype=torch.long,
        device=device,
    )

    gen_start_idx = enc_user_prompt.shape[1]

    for _ in range(max_token):
        output_logits = model(enc_user_prompt[:, -context_len:])
        output_logits = output_logits[0, -1, :]
        generated_ids = enc_user_prompt[0].tolist()
        output_logits = apply_repetition_penalty(
            output_logits,
            generated_ids,
            penalty=repetition_penalty,
        )
        output_logits = apply_no_repeat_ngram_mask(
            output_logits,
            generated_ids,
            ngram_size=no_repeat_ngram_size,
        )

        if ban_token_ids is not None:
            output_logits = output_logits.clone()
            for token_id in ban_token_ids:
                if 0 <= token_id < output_logits.numel() and token_id != end_token:
                    output_logits[token_id] = float("-inf")

        if not do_sample:
            gen_token = torch.argmax(output_logits)
        else:
            if temperature <= 0:
                raise ValueError("temperature must be > 0 when do_sample=True")
            output_logits = output_logits / temperature

            if top_k is not None and top_k > 0:
                kth = torch.topk(
                    output_logits,
                    k=min(top_k, output_logits.numel()),
                ).values[-1]
                output_logits = output_logits.masked_fill(
                    output_logits < kth,
                    float("-inf"),
                )

            probs = run_softmax(output_logits, -1)
            probs = top_p_sampling(probs, top_p)

            dist = Categorical(probs=probs)
            gen_token = dist.sample()

        if gen_token.item() == end_token:
            break

        enc_user_prompt = torch.cat(
            [
                enc_user_prompt,
                gen_token.view(1, 1),
            ],
            dim=1,
        )

    return enc_user_prompt[0][gen_start_idx:]






def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

import os


import os
import gc
import regex as re
from collections import Counter, defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries_by_whitespace(
    file,
    chunk_size_bytes: int,
    max_scan_bytes: int = 1024 * 1024,
) -> list[int]:
    """
    按固定大小切文件，但是把边界移动到附近的空白字符上。

    注意：
    边界停在 whitespace 的位置，而不是 whitespace 后面。
    这样下一块会以空格开头，能保留 GPT-style pre-tokenization 的前导空格。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    boundaries = [0]
    pos = chunk_size_bytes

    whitespace = set(b" \n\r\t")

    while pos < file_size:
        file.seek(pos)

        found = None
        scanned = 0

        while scanned < max_scan_bytes and pos + scanned < file_size:
            b = file.read(1)
            if not b:
                break

            if b[0] in whitespace:
                found = pos + scanned
                break

            scanned += 1

        if found is None:
            # 极端情况下附近没有空白，就硬切
            found = min(pos, file_size)

        if found <= boundaries[-1]:
            found = min(boundaries[-1] + chunk_size_bytes, file_size)

        boundaries.append(found)
        pos = found + chunk_size_bytes

    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    return boundaries


def _split_by_special_tokens(text: str, special_tokens: list[str]):
    """
    把 special token 从文本中切掉。
    返回不包含 special token 的普通文本片段。
    """
    if not special_tokens:
        yield text
        return

    segments = [text]

    for st in special_tokens:
        new_segments = []
        for seg in segments:
            if seg:
                new_segments.extend(seg.split(st))
        segments = new_segments

    for seg in segments:
        if seg:
            yield seg


def _count_pretokens_chunk(
    input_path: str ,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter:
    counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="ignore")
    del raw

    for segment in _split_by_special_tokens(text, special_tokens):
        for m in re.finditer(PAT, segment):
            counter[m.group()] += 1

    return counter


def _python_bpe_merge(
    word_freq: Counter,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    Python fallback 版本。
    大数据上会慢，优先建议使用 C++ bpe_merge。
    """
    token_to_id = {bytes([i]): i for i in range(256)}
    id_to_token = {i: bytes([i]) for i in range(256)}

    merge_times = vocab_size - 256 - len(special_tokens)
    next_token_id = 256
    merges = []

    words = list(word_freq.keys())
    word_freqs = [word_freq[w] for w in words]

    word_tokens = []
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    for word_id, word in enumerate(words):
        freq = word_freqs[word_id]
        tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
        word_tokens.append(tokens)

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)

    print("===== initial pair counts done =====")
    print("num pairs:", len(pair_counts))

    for merge_step in range(merge_times):
        if not pair_counts:
            break

        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda x: (x[1], x[0]),
        )

        new_token = best_pair[0] + best_pair[1]

        merges.append(best_pair)

        token_to_id[new_token] = next_token_id
        id_to_token[next_token_id] = new_token
        next_token_id += 1

        affected_word_ids = list(pair_to_words.get(best_pair, set()))

        for word_id in affected_word_ids:
            old_tokens = word_tokens[word_id]
            freq = word_freqs[word_id]

            if len(old_tokens) < 2:
                continue

            # 删除旧 pair 贡献
            for i in range(len(old_tokens) - 1):
                old_pair = (old_tokens[i], old_tokens[i + 1])
                pair_counts[old_pair] -= freq

                if pair_counts[old_pair] <= 0:
                    pair_counts.pop(old_pair, None)
                    pair_to_words.pop(old_pair, None)
                else:
                    if old_pair in pair_to_words:
                        pair_to_words[old_pair].discard(word_id)
                        if not pair_to_words[old_pair]:
                            pair_to_words.pop(old_pair, None)

            # 合并 best_pair
            new_tokens = []
            i = 0

            while i < len(old_tokens):
                if (
                    i < len(old_tokens) - 1
                    and old_tokens[i] == best_pair[0]
                    and old_tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1

            new_tokens = tuple(new_tokens)
            word_tokens[word_id] = new_tokens

            # 添加新 pair 贡献
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(word_id)

        if merge_step % 100 == 0:
            print(
                f"merge {merge_step}/{merge_times}, "
                f"best_count={best_count}, "
                f"num_pairs={len(pair_counts)}"
            )

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in token_to_id:
            token_to_id[token_bytes] = next_token_id
            id_to_token[next_token_id] = token_bytes
            next_token_id += 1

    vocab = {idx: tok for idx, tok in id_to_token.items()}

    return vocab, merges


def _cpp_bpe_merge(
    word_freq: Counter,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = True,
):
    """
    使用 C++ 扩展 bpe_merge。
    需要你已经成功编译：

        import bpe_merge

    C++ 侧函数名应为：

        bpe_merge.bpe_train_core(...)
    """
    import bpe_merge

    words = []
    freqs = []

    for word, freq in word_freq.items():
        words.append(word.encode("utf-8"))
        freqs.append(int(freq))

    del word_freq
    gc.collect()

    num_merges = vocab_size - 256 - len(special_tokens)

    print("===== calling C++ merge core =====")
    print("num words:", len(words))
    print("num merges:", num_merges)

    merge_id_pairs = bpe_merge.bpe_train_core(
        words,
        freqs,
        num_merges,
        256,
        verbose,
    )

    print("===== C++ merge done =====")

    del words
    del freqs
    gc.collect()

    id_to_token = {i: bytes([i]) for i in range(256)}
    token_to_id = {bytes([i]): i for i in range(256)}

    merges = []
    next_id = 256

    for left_id, right_id in merge_id_pairs:
        left_bytes = id_to_token[int(left_id)]
        right_bytes = id_to_token[int(right_id)]

        new_bytes = left_bytes + right_bytes

        merges.append((left_bytes, right_bytes))

        id_to_token[next_id] = new_bytes
        token_to_id[new_bytes] = next_id

        next_id += 1

    for token in special_tokens:
        token_bytes = token.encode("utf-8")

        if token_bytes not in token_to_id:
            token_to_id[token_bytes] = next_id
            id_to_token[next_id] = token_bytes
            next_id += 1

    vocab = {idx: tok for idx, tok in id_to_token.items()}

    return vocab, merges


def run_train_bpe(
    input_path: str ,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    OWT 友好版本 BPE trainer。

    推荐 OWT 调用方式：

        vocab, merges = run_train_bpe(
            "./data/owt_train.txt",
            32000,
            ["<|endoftext|>"],
            chunk_size_mb=32,
            min_freq=2,
            max_pretokens=1_000_000,
            use_cpp=True,
            verbose=True,
        )

    如果你想严格跑小测试，可以用：

        min_freq=1,
        max_pretokens=None,
        use_cpp=False
    """

    chunk_size_mb = kwargs.get("chunk_size_mb", 32)
    min_freq = kwargs.get("min_freq", 2)
    max_pretokens = kwargs.get("max_pretokens", 1_000_000)
    use_cpp = kwargs.get("use_cpp", True)
    verbose = kwargs.get("verbose", True)

    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    word_freq = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries_by_whitespace(
            f,
            chunk_size_bytes=chunk_size_bytes,
        )

    num_chunks = len(boundaries) - 1
    print(f"num chunks: {num_chunks}")

    for chunk_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end <= start:
            continue

        chunk_counter = _count_pretokens_chunk(
            input_path=input_path,
            start=start,
            end=end,
            special_tokens=special_tokens,
        )

        word_freq.update(chunk_counter)

        del chunk_counter

        if chunk_idx % 10 == 0:
            print(
                f"processed chunk {chunk_idx}/{num_chunks}, "
                f"unique pretokens={len(word_freq)}"
            )

    print("===== splitting done =====")
    print("unique pretokens before filter:", len(word_freq))

    if min_freq is not None and min_freq > 1:
        word_freq = Counter({
            w: c
            for w, c in word_freq.items()
            if c >= min_freq
        })
        gc.collect()

    print("unique pretokens after min_freq:", len(word_freq))

    if max_pretokens is not None and len(word_freq) > max_pretokens:
        word_freq = Counter(dict(word_freq.most_common(max_pretokens)))
        gc.collect()

    print("unique pretokens after max_pretokens:", len(word_freq))

    if use_cpp:
        try:
            return _cpp_bpe_merge(
                word_freq=word_freq,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                verbose=verbose,
            )
        except ImportError as e:
            print("WARNING: failed to import bpe_merge, falling back to Python BPE.")
            print(e)

    return _python_bpe_merge(
        word_freq=word_freq,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )



from collections.abc import Iterable, Iterator
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import pickle
import regex as re

def save_with_pickle(data, filename):
    """使用pickle格式保存字典"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"字典已保存为pickle文件: {filename}")

def load_with_pickle(filename):
    """从pickle文件加载字典"""
    with open(filename, 'rb') as f:
        return pickle.load(f)



import regex as re

def _pre_tokenize(text: str, special_tokens, PAT):
    if special_tokens is None or not special_tokens:
        return re.findall(PAT, text)
        
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped = [re.escape(st) for st in special_tokens]
    parts = re.split('(' + '|'.join(escaped) + ')', text)
    
    ret = []
    for part in parts:
        if part in special_tokens:
            ret.append(part)
        elif part:
            ret.extend(re.findall(PAT, part))
    return ret

def find_first_key_by_value(d, value):
    return next((k for k, v in d.items() if v == value), None)


def _merge(word, merges, vocab):
    while True:
        merged = False
        for p in merges:
            i = 0
            while i < len(word) - 1:
                if word[i] == p[0] and word[i + 1] == p[1]:
                    word = word[: i] + [word[i] + word[i + 1]] + word[i + 2: ]
                    merged = True
                    break
                i += 1
            if merged:
                break
        if not merged:
            break
    word = [find_first_key_by_value(vocab, token) for token in word]
    return word



class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.cache = {}

    def encode(self, text: str) -> list[int]:
        text_ls = _pre_tokenize(text, self.special_tokens, self.PAT)
        ret = []

        for tl in text_ls:
            if self.special_tokens and tl in self.special_tokens:
                ret.append(self.vocab_inv[tl.encode("utf-8")])
            else:
                tl_bytes = tl.encode("utf-8")

                if tl_bytes in self.cache:
                    ret.extend(self.cache[tl_bytes])
                else:
                    seg = _merge(
                        [bytes([b]) for b in tl_bytes],
                        self.merges,
                        self.vocab,
                    )
                    if len(self.cache) < 1_000_000:
                        self.cache[tl_bytes] = seg
                    ret.extend(seg)

        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for tidx in token_ids:
                yield tidx

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return full_bytes.decode("utf-8", errors="replace")





import os
import numpy as np
from typing import Tuple


def make_npy(tk: Tokenizer, inputPath: tuple[str, str], outputPath: str, lines: int):
    import numpy as np
    import os

    shard_size = 50_000_000

    def encode_and_save(path, split_name):
        ids = []
        shard_id = 0

        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            if lines == -1:
                iterator = enumerate(file)
            else:
                iterator = ((i, file.readline()) for i in range(lines))

            for i, line in iterator:
                if not line:
                    break

                ids.extend(tk.encode(line))

                if i % 10000 == 0:
                    print(
                        f"{split_name} lines: {i}, "
                        f"current shard tokens: {len(ids)}, "
                        f"shard: {shard_id}"
                    )

                if len(ids) >= shard_size:
                    arr = np.array(ids, dtype=np.uint16)
                    out_file = f"{outputPath}-{split_name}-{lines}-shard{shard_id}.npy"
                    np.save(out_file, arr)

                    print(f"saved {out_file}, tokens={len(ids)}")

                    ids.clear()
                    shard_id += 1

            if ids:
                arr = np.array(ids, dtype=np.uint16)
                out_file = f"{outputPath}-{split_name}-{lines}-shard{shard_id}.npy"
                np.save(out_file, arr)

                print(f"saved {out_file}, tokens={len(ids)}")

        print(f"===={split_name} completed====")

    encode_and_save(inputPath[0], "train")
    encode_and_save(inputPath[1], "valid")

    print("====done====")





# =============================================================

"""
下面的代码由 AI 代写, 是为了快速实现训练集的 Tokenization 而榨干多核 CPU 定制的
实测效果多核综合 CPU 占用率稳居 98% 以上, CPU 温度稳居 90 以上, 比原先的纯 python 单核版本快了 41538 %
同样由 AI 所写的还有 bpe_fast.cpp 及配套的 setup_bpe_fast.py
"""

# =============================================================



import os
import regex as re
import numpy as np
import multiprocessing as mp
from collections.abc import Iterable, Iterator


def _pre_tokenize_high_performance(text: str, special_tokens, PAT):
    if special_tokens is None or not special_tokens:
        return re.findall(PAT, text)

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped = [re.escape(st) for st in special_tokens]
    parts = re.split("(" + "|".join(escaped) + ")", text)

    ret = []

    for part in parts:
        if part in special_tokens:
            ret.append(part)
        elif part:
            ret.extend(re.findall(PAT, part))

    return ret


def _merge_python_fast_fallback(piece_bytes: bytes, merge_ranks, vocab_inv):
    """
    Python fallback，C++ 不可用时使用。
    比你原来的 _merge 快很多，因为它不是每次扫全部 merges。
    """
    tokens = [bytes([b]) for b in piece_bytes]

    if len(tokens) <= 1:
        return [vocab_inv[tokens[0]]] if tokens else []

    while True:
        best_rank = None
        best_pair = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merge_ranks.get(pair)

            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_pair = pair

        if best_pair is None:
            break

        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        tokens = new_tokens

        if len(tokens) <= 1:
            break

    return [vocab_inv[t] for t in tokens]


class FastTokenizerOWTHighPerformance:
    """
    OWT 高性能 tokenizer。

    特点：
    1. regex pre-tokenize 仍然在 Python。
    2. BPE merge 默认使用 C++ bpe_fast。
    3. 带 cache。
    4. 可用于 make_npy_owt_high_performance 多进程。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
        cache_size: int = 1_000_000,
        use_cpp: bool = True,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.cache_size = cache_size

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.special_token_bytes_to_id = {
            st.encode("utf-8"): self.vocab_inv[st.encode("utf-8")]
            for st in self.special_tokens
            if st.encode("utf-8") in self.vocab_inv
        }

        self.cache = {}

        self.merge_ranks = {
            pair: i
            for i, pair in enumerate(merges)
        }

        self.use_cpp = False
        self.cpp_encoder = None

        if use_cpp:
            try:
                import bpe_fast

                vocab_items = [
                    (int(idx), tok)
                    for idx, tok in vocab.items()
                ]

                self.cpp_encoder = bpe_fast.FastBPEEncoder(
                    vocab_items,
                    merges,
                )

                self.use_cpp = True
                print("FastTokenizerOWTHighPerformance: using C++ bpe_fast")

            except ImportError as e:
                print("FastTokenizerOWTHighPerformance: C++ bpe_fast not found, using Python fallback")
                print(e)

    def encode(self, text: str) -> list[int]:
        text_ls = _pre_tokenize_high_performance(
            text,
            self.special_tokens,
            self.PAT,
        )

        ret = []

        for tl in text_ls:
            if self.special_tokens and tl in self.special_tokens:
                token_bytes = tl.encode("utf-8")
                ret.append(self.special_token_bytes_to_id[token_bytes])
                continue

            tl_bytes = tl.encode("utf-8")

            cached = self.cache.get(tl_bytes)
            if cached is not None:
                ret.extend(cached)
                continue

            if self.use_cpp:
                seg = list(self.cpp_encoder.encode_piece(tl_bytes))
            else:
                seg = _merge_python_fast_fallback(
                    tl_bytes,
                    self.merge_ranks,
                    self.vocab_inv,
                )

            if len(self.cache) < self.cache_size:
                self.cache[tl_bytes] = seg

            ret.extend(seg)

        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for x in ids:
                yield x

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[int(token_id)] for token_id in ids)
        return full_bytes.decode("utf-8", errors="replace")


_FAST_TK_HP = None


def _init_fast_tokenizer_worker_high_performance(
    vocab,
    merges,
    special_tokens,
    cache_size,
    use_cpp,
):
    global _FAST_TK_HP

    _FAST_TK_HP = FastTokenizerOWTHighPerformance(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        cache_size=cache_size,
        use_cpp=use_cpp,
    )


def _encode_lines_worker_high_performance(lines):
    global _FAST_TK_HP

    ids = []

    for line in lines:
        ids.extend(_FAST_TK_HP.encode(line))

    return ids


def _line_batches_high_performance(path: str, lines: int, batch_lines: int):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        batch = []

        if lines == -1:
            for line in f:
                batch.append(line)

                if len(batch) >= batch_lines:
                    yield batch
                    batch = []
        else:
            for i, line in enumerate(f):
                if i >= lines:
                    break

                batch.append(line)

                if len(batch) >= batch_lines:
                    yield batch
                    batch = []

        if batch:
            yield batch


def make_npy_owt_high_performance(
    tk: FastTokenizerOWTHighPerformance,
    inputPath: tuple[str, str],
    outputPath: str,
    lines: int,
    shard_size: int = 50_000_000,
    batch_lines: int = 2048,
    num_workers: int   = None,
    use_cpp: bool = True,
):
    """
    OWT 高性能版本 make_npy。

    参数：
        tk:
            FastTokenizerOWTHighPerformance 实例。

        inputPath:
            (train_txt, valid_txt)

        outputPath:
            输出前缀，例如 "./data/owt"

        lines:
            -1 表示全量，否则只处理前 lines 行。

        shard_size:
            每个 npy shard 多少 token。
            50_000_000 个 uint16 token 大约 100MB。

        batch_lines:
            每个 worker 一次处理多少行。
            太小：进程通信开销大。
            太大：内存峰值高。
            推荐 1024 / 2048 / 4096。

        num_workers:
            默认 os.cpu_count() - 1。

        use_cpp:
            worker 内是否使用 C++ bpe_fast。
    """

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    if len(tk.vocab) <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32

    print("==== make_npy_owt_high_performance ====")
    print("num_workers:", num_workers)
    print("batch_lines:", batch_lines)
    print("shard_size:", shard_size)
    print("dtype:", dtype)
    print("use_cpp:", use_cpp)

    vocab = tk.vocab
    merges = tk.merges
    special_tokens = tk.special_tokens
    cache_size = tk.cache_size

    def encode_split(path: str, split_name: str):
        print(f"==== encoding {split_name} ====")

        shard_id = 5
        total_tokens = 0
        buffer = []
        buffer_tokens = 0
        processed_batches = 0

        pool = mp.Pool(
            processes=num_workers,
            initializer=_init_fast_tokenizer_worker_high_performance,
            initargs=(
                vocab,
                merges,
                special_tokens,
                cache_size,
                use_cpp,
            ),
        )

        try:
            batches = _line_batches_high_performance(
                path=path,
                lines=lines,
                batch_lines=batch_lines,
            )

            for ids in pool.imap(
                _encode_lines_worker_high_performance,
                batches,
                chunksize=1,
            ):
                if ids:
                    arr = np.asarray(ids, dtype=dtype)
                    buffer.append(arr)
                    buffer_tokens += arr.shape[0]
                    total_tokens += arr.shape[0]

                processed_batches += 1

                if processed_batches % 10 == 0:
                    print(
                        f"{split_name}: batches={processed_batches}, "
                        f"current_shard_tokens={buffer_tokens}, "
                        f"total_tokens={total_tokens}, "
                        f"shard={shard_id}"
                    )

                if buffer_tokens >= shard_size:
                    out_file = f"{outputPath}-{split_name}-{lines}-shard{shard_id}.npy"

                    shard_arr = np.concatenate(buffer, axis=0)
                    np.save(out_file, shard_arr)

                    print(
                        f"saved {out_file}, "
                        f"tokens={shard_arr.shape[0]}, "
                        f"total_tokens={total_tokens}"
                    )

                    buffer.clear()
                    buffer_tokens = 0
                    shard_id += 1

            if buffer_tokens > 0:
                out_file = f"{outputPath}-{split_name}-{lines}-shard{shard_id}.npy"

                shard_arr = np.concatenate(buffer, axis=0)
                np.save(out_file, shard_arr)

                print(
                    f"saved {out_file}, "
                    f"tokens={shard_arr.shape[0]}, "
                    f"total_tokens={total_tokens}"
                )

                buffer.clear()
                buffer_tokens = 0

        finally:
            pool.close()
            pool.join()

        print(f"==== {split_name} completed, total_tokens={total_tokens} ====")

    encode_split(inputPath[0], "train")
    encode_split(inputPath[1], "valid")

    print("==== make_npy_owt_high_performance done ====")
