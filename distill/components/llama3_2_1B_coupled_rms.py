# The code in this file implements the LLaMA 3.2 1B and 3B models with CoupledNorm as nomralizer
# This file is based on the following files:
# https://github.com/meta-pytorch/torchtune/blob/main/torchtune/models/llama3_2/_component_builders.py
# https://github.com/meta-pytorch/torchtune/blob/main/torchtune/models/llama3_2/_model_builders.py
# https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/transformer.py
# https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/rms_norm.py

# BSD 3-Clause "New" or "Revised" License
# License: https://github.com/meta-pytorch/torchtune/blob/main/LICENSE

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file at https://github.com/meta-pytorch/torchtune/blob/main/LICENSE
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Dict, List, Optional, Any
from functools import partial

import torch
import torch.nn as nn

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import TiedLinear
from torchtune.modules import MultiHeadAttention, TransformerDecoder
from torchtune.modules import MultiHeadAttention
from torchtune.models.llama3_2._component_builders import lora_llama3_2_self_attention, llama3_mlp, lora_llama3_mlp
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, scaler=None, return_scaler=False):
        input_dtype = x.dtype
        x = x.float()
        if scaler is None:
            scaler = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        y = (self.scale * x * scaler).to(input_dtype)
        return (y, scaler) if return_scaler else y


class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
    """

    def __init__(
        self,
        idx: int,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
        norm_normal_before: int | None = None,
    ) -> None:
        super().__init__()
        self.idx = idx
        self.norm_normal_before = norm_normal_before
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[Any] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h, attn_scaler = self.sa_norm(x, return_scaler=True)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        if self.norm_normal_before is not None and self.idx <= self.norm_normal_before:
            attn_scaler = None

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h, scaler=attn_scaler))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


def llama3_2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 32,
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3.2 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        scale_factor (int): scaling factor for RoPE. Default: 32
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)

    layers = nn.ModuleList()
    for i in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            idx=i,
            norm_normal_before=norm_normal_before,
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def lora_llama3_2(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # llama3.2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    intermediate_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 500_000,
    scale_factor: int = 32,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None,
) -> TransformerDecoder:
    """
    Return a version of Llama3.2 (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied based on the passed in configuration.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        scale_factor (int): scaling factor for RoPE. Default: 32
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 model with LoRA applied to
        a subset of the attention projections in each layer.

    """

    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    head_dim = embed_dim // num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    
    layers = nn.ModuleList()
    for i in range(num_layers):
        self_attn = lora_llama3_2_self_attention(
            lora_modules=lora_attn_modules,
            pos_embeddings=rope,
            head_dim=head_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
        )

        if apply_lora_to_mlp:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
        else:
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base)

        layer = TransformerSelfAttentionLayer(
            idx=i,
            norm_normal_before=norm_normal_before,
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
        
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    if apply_lora_to_output:
        raise ValueError(
            "apply_lora_to_output is currently not supporting in llama3.2 1b and 3b,"
            "as the projection layer weights are tied to the embeddings"
        )
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return model


"""
Model builders build specific instantiations using component builders. For example
the llama3_2_1b model builder uses the llama3_2 component builder to create the
Llama3.2 1B model.
"""
def llama3_2_1b(
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.
    
    Args:
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        tie_word_embeddings=tie_word_embeddings,
        norm_normal_before=norm_normal_before
    )

def llama3_2_3b(
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 3b parameter values.

    Args:
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 3B model
    """
    return llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        tie_word_embeddings=tie_word_embeddings,
        norm_normal_before=norm_normal_before,
    )

def lora_llama3_2_1b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 1B model with LoRA enabled.
    The Llama3.2 defaults are the same as in :func:`~torchtune.models.llama3_2.llama3_2_1b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.
    
    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model with LoRA applied
    """
    return lora_llama3_2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        tie_word_embeddings=tie_word_embeddings,
        norm_normal_before=norm_normal_before
    )

def lora_llama3_2_3b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    tie_word_embeddings: bool = True,
    norm_normal_before: int | None = None
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 3B model with LoRA enabled.
    The Llama3.2 defaults are the same as in :func:`~torchtune.models.llama3_2.llama3_2_3b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 3B model with LoRA applied
    """
           
    return lora_llama3_2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        tie_word_embeddings=tie_word_embeddings,
        norm_normal_before=norm_normal_before
    )
