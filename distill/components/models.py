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
from torchtune.models.llama3_2._component_builders import lora_llama3_2_self_attention, llama3_mlp # , lora_llama3_mlp
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules import FeedForward
from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear


def setup_caches(
    self,
    batch_size: int,
    dtype: torch.dtype,
    *,
    encoder_max_seq_len: Optional[int] = None,
    decoder_max_seq_len: Optional[int] = None,
):
    has_decoder_layers = True
    if has_decoder_layers:
        if decoder_max_seq_len is not None:
            self.decoder_max_cache_seq_len = decoder_max_seq_len
        else:
            self.decoder_max_cache_seq_len = self.max_seq_len
    for layer in self.layers:
        layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=self.encoder_max_cache_seq_len,
            decoder_max_seq_len=self.decoder_max_cache_seq_len,
        )
# Mokey patch, because original implementaion checks for class that we've redefined
TransformerDecoder.setup_caches = setup_caches

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, unfreeze = True, coupled_downscale: float | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.unfreeze = unfreeze

    def forward(self, x, scaler=None, return_scaler=False):
        input_dtype = x.dtype
        x = x.float()
        if scaler is None:
            scaler = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        y = (self.scale * x * scaler).to(input_dtype)
        return (y, scaler) if return_scaler else y

    # Make RMSNorm weight learnable
    def adapter_params(self) -> List[str]:
        if not self.unfreeze: return []
        adapter_params = ["scale"]
        return adapter_params


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
        norm_normal_after: int | None = None,
        coupled_init_downscale: float = 0.9,
    ) -> None:
        super().__init__()
        self.idx = idx
        self.norm_normal_before = norm_normal_before
        self.norm_normal_after = norm_normal_after
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()
        self.coupled_init_downscale = coupled_init_downscale

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
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h, attn_scaler = self.sa_norm(x, return_scaler=True)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)

        # fix_scalers = [0.845, 0.817, 0.857, 0.892, 0.93, 0.933, 0.939, 0.916, 0.923, 0.875, 0.971, 0.907, 0.915, 0.91, 0.871, 0.925]

        # print(self.norm_normal_before, self.norm_normal_after, self.idx <= self.norm_normal_before or self.idx >= self.norm_normal_after)
        if (self.idx <= self.norm_normal_before or self.idx >= self.norm_normal_after):
            attn_scaler = None
        else:
            attn_scaler = attn_scaler * self.coupled_init_downscale

        # attn_scaler = None

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h, scaler=attn_scaler))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out

def lora_llama3_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> FeedForward:
    adapter_cls = DoRALinear if use_dora else LoRALinear
    gate_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    # gate_proj = nn.Linear(dim, hidden_dim, bias=False)

    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )

    # down_proj = nn.Linear(hidden_dim, dim, bias=False)

    up_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )

    # up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=up_proj,
    )



class FeedForwardFineuneUp(nn.Module):
    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h

    def adapter_params(self) -> List[str]:
        return ["w1.weight", "w2.weight", "w3.weight"] # 

def llama3_mlp_finetune(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForwardFineuneUp(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


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
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    head_dim = embed_dim // num_heads
    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor)
    
    layers = nn.ModuleList()
    for i in range(num_layers):
        if len(lora_attn_modules) > 0:
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
        else:
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

        if apply_lora_to_mlp and i > norm_normal_before and i < norm_normal_after:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
            # mlp = llama3_mlp_finetune(dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base)
        else:
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base)


        layer = TransformerSelfAttentionLayer(
            idx=i,
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps, unfreeze=False),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps, unfreeze=i > norm_normal_before and i < norm_normal_after, coupled_downscale=coupled_init_downscale),
            norm_normal_before=norm_normal_before,
            norm_normal_after=norm_normal_after,
            coupled_init_downscale=coupled_init_downscale,
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
        norm=RMSNorm(embed_dim, eps=norm_eps, unfreeze=False),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return model

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
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
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
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
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
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
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
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
    )


from typing import List, Optional

from torchtune.models.qwen3._component_builders import lora_qwen3, lora_qwen3_mlp, qwen3_mlp
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

from functools import partial
from typing import Optional
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torch import nn
from torchtune.modules.transformer import TransformerDecoder
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings
from torchtune.models.qwen3._attention import Qwen3Attention
from torchtune.modules import FeedForward, TiedLinear
from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear
from torchtune.modules import RMSNorm as RMSNormNormal



def lora_qwen3(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # qwen2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    q_proj_bias: bool = True,
    k_proj_bias: bool = True,
    v_proj_bias: bool = True,
    q_norm: bool = False,
    k_norm: bool = False,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    head_dim = head_dim or embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    layers = nn.ModuleList()
    for i in range(num_layers):
        self_attn = Qwen3Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=q_proj_bias),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=k_proj_bias),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=v_proj_bias),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            q_norm=RMSNormNormal(dim=head_dim, eps=norm_eps) if q_norm else None, # norm on head_dim
            k_norm=RMSNormNormal(dim=head_dim, eps=norm_eps) if k_norm else None,
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        if apply_lora_to_mlp and i > norm_normal_before and i < norm_normal_after:
            mlp = lora_qwen3_mlp(
                dim=embed_dim,
                hidden_dim=intermediate_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                use_dora=use_dora,
                lora_dropout=lora_dropout,
            )
        else:
            mlp = qwen3_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

        layer = TransformerSelfAttentionLayer(
            idx=i,
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps, unfreeze=False),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps, unfreeze=i > norm_normal_before and i < norm_normal_after, coupled_downscale=coupled_init_downscale),
            norm_normal_before=norm_normal_before,
            norm_normal_after=norm_normal_after,
            coupled_init_downscale=coupled_init_downscale,
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    if tie_word_embeddings:
        if apply_lora_to_output:
            raise ValueError(
                "apply_lora_to_output is incompatible with tie_word_embeddings,"
                " as there would be no output to apply lora to!"
            )
        output_proj = TiedLinear(tok_embeddings)
    else:
        # TODO: quantize_base is not applied to final output_proj currently.
        adapter_cls = DoRALinear if use_dora else LoRALinear
        output_proj = (
            adapter_cls(embed_dim, vocab_size, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            if apply_lora_to_output
            else nn.Linear(embed_dim, vocab_size, bias=False)
        )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps, unfreeze=False),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to higher precision, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                # TODO this is clowny, figure out a better way to get what precision the rest
                # of the model is in
                dtype=tok_embeddings.weight.dtype,
                offload_to_cpu=True,
            )
        )

    return model


def lora_qwen3_0_6b_instruct(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
    )

def lora_qwen3_1_7b_instruct(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
    )

def lora_qwen3_4b_instruct(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
    )

def lora_qwen3_8b_instruct(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    norm_normal_before: int | None = None,
    norm_normal_after: int | None = None,
    coupled_init_downscale: float = 0.9,
) -> TransformerDecoder:
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        norm_normal_before=norm_normal_before,
        norm_normal_after=norm_normal_after,
        coupled_init_downscale=coupled_init_downscale,
    )