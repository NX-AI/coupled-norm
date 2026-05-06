import torch
import torch.nn as nn
import types
from torch.library import custom_op

torch._inductor.config.epilogue_fusion = True
torch._inductor.config.cpp_wrapper = True

@custom_op("inplace_addmm::inplace_addmm", mutates_args=("bias",))
def inplace_addmm(bias: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    torch.ops.aten.addmm_.default(bias, x, weight)
    # torch.addmm_(bias, x, weight)

@inplace_addmm.register_fake
def inplace_addmm_fake_(bias: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> None:
    torch.ops.aten.addmm_.default(bias, x, weight)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, weight=None, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.ones(hidden_size,))
        self.eps = eps

    def forward(self, x, scaler=None, return_scaler=False):
        input_dtype = x.dtype
        if scaler is None:
            var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps
            scaler = torch.rsqrt(var)

        y = self.weight * (x * scaler).to(input_dtype)
        return (y, scaler) if return_scaler else y

def patch_coupled_model_manual(model, coupled_downscale, no_coupled_before, no_coupled_after):
    for i, block in enumerate(model.model.layers):
        if no_coupled_before < i and i < no_coupled_after:
            block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
            block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)

            def forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
                **kwargs,
            ) -> torch.Tensor:
                residual = hidden_states
                hidden_states, norm_scaler = self.input_layernorm(hidden_states, return_scaler=True)
                # Self Attention
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states, scaler=coupled_downscale * norm_scaler)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                return hidden_states

            block.forward = types.MethodType(forward, block)


def coupled_config_from_hf(model):
    config = model.config
    
    if not hasattr(config, "coupled_downscale"):
        print("This model doesn't have a CoupledNorm config, running with couplednorm applied on all layers and no downscale!")
        return 1.0, -1, 100

    coupled_downscale = config.coupled_downscale if config.coupled_downscale is not None else 1.0
    no_coupled_before = config.no_coupled_before if config.no_coupled_before is not None else -1
    no_coupled_after = config.no_coupled_after if config.no_coupled_after is not None else 100
    return coupled_downscale, no_coupled_before, no_coupled_after


def patch_coupled_model_hf(model, merged=False, use_inplace_addmm=True, merge_up_proj=True):
    coupled_downscale, no_coupled_before, no_coupled_after = coupled_config_from_hf(model)

    if merged:
        patch_coupled_merged(model, coupled_downscale, no_coupled_before, no_coupled_after, use_inplace_addmm, merge_up_proj)
    else:
        patch_coupled_model_manual(model, coupled_downscale, no_coupled_before, no_coupled_after)


"""
This fuction merges all operations from the second apply normalization step into other components, in a way that no kernel for normalization will be generated:
- Adddition of the residuals are done in the output projection of the attention layer
- The affinewise weights of the layernorm are statically merged into the up projection weights of the MLP
- The scaling of the normalization is merged into the silu kernel after the up projection, as that is mathematically equivilant to applying it before.
"""
def patch_coupled_merged(model, coupled_downscale, no_coupled_before, no_coupled_after, use_inplace_addmm=True, merge_up_proj=False):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as apply_rotary_pos_emb_qwen, eager_attention_forward as eager_attention_forward_qwen
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama, eager_attention_forward as eager_attention_forward_llama

    for i, block in enumerate(model.model.layers):
        if no_coupled_before < i and i < no_coupled_after:
            print(f"Patch {i}")

            block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
            # block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)
            
            # Merge affine weights and coupled_downscale factor into upprojection of MLP layer
            ln_affine_weight_to_merge = coupled_downscale * block.post_attention_layernorm.weight.data.unsqueeze(0)
            block.mlp.up_proj.weight.data = (block.mlp.up_proj.weight.data * ln_affine_weight_to_merge).contiguous().clone()
            block.mlp.gate_proj.weight.data = (block.mlp.gate_proj.weight.data * ln_affine_weight_to_merge).contiguous().clone()

            if merge_up_proj:
                block.mlp.up_gate_proj = nn.Linear(model.config.hidden_size, 2 * model.config.intermediate_size, bias=False)
                block.mlp.up_gate_proj.weight.data = torch.cat((block.mlp.up_proj.weight.data, block.mlp.gate_proj.weight.data), dim=0).contiguous().clone()
    
                block.self_attn.qkv_proj = nn.Linear(model.config.hidden_size, (model.config.num_attention_heads + 2 * model.config.num_key_value_heads) * model.config.head_dim, bias=False)
                block.self_attn.qkv_proj.weight.data = torch.cat((block.self_attn.q_proj.weight.data, block.self_attn.k_proj.weight.data, block.self_attn.v_proj.weight.data), dim=0).contiguous().clone()

            block.self_attn.o_proj.weight.data = block.self_attn.o_proj.weight.data.t().contiguous().clone()
            block.mlp.down_proj.weight.data = block.mlp.down_proj.weight.data.t().contiguous().clone()

            def baddmm(bias: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
                out_shape = bias.shape
                bias = bias.flatten(0, 1)
                x = x.flatten(0, 1)
                if use_inplace_addmm:
                    inplace_addmm(bias, x, weight)
                else:
                    torch.addmm_(bias, x, weight)
                return bias.view(out_shape)

            def forward_attn_qwen(
                self,
                residual,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                if merge_up_proj:
                    qkv = self.qkv_proj(hidden_states)
                    q, k, v = qkv.split([self.config.num_attention_heads * self.config.head_dim, self.config.num_key_value_heads * self.config.head_dim, self.config.num_key_value_heads * self.config.head_dim], dim=-1)
                    query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
                    key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
                    value_states = v.view(hidden_shape).transpose(1, 2)
                else:
                    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_qwen(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward_qwen
                )

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,  # diff with Llama
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                residual_out = baddmm(residual, attn_output, self.o_proj.weight)
                return residual_out, attn_weights

            def forward_attn_llama(
                self,
                residual,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                **kwargs,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb_llama(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward_llama
                )

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                residual_out = baddmm(residual, attn_output, self.o_proj.weight)
                return residual_out, attn_weights

            def forward_mlp(self, x, residual, scaler):
                gate = (scaler * self.gate_proj(x)).to(x.dtype)
                up = (scaler * self.up_proj(x)).to(x.dtype)
                hid = self.act_fn(gate) * up
                return baddmm(residual, hid, self.down_proj.weight)
                # return self.down_proj(self.act_fn(gate) * up) + residual
            
            def forward_mlp_merged(self, x, residual, scaler):
                up_gate_norm = (scaler * self.up_gate_proj(x)).to(x.dtype)
                up, gate = up_gate_norm.split([self.config.intermediate_size, self.config.intermediate_size], dim=-1)
                hid = self.act_fn(gate) * up
                return baddmm(residual, hid, self.down_proj.weight)
                # return self.down_proj(self.act_fn(gate) * up) + residual

            model_type = model.config.model_type
            if model_type == "qwen3":
                block.self_attn.forward = types.MethodType(forward_attn_qwen, block.self_attn)
            elif model_type == "llama":
                block.self_attn.forward = types.MethodType(forward_attn_llama, block.self_attn)
            else:
                raise Exception(f"Merging isn't supported for {model_type}")

            if merge_up_proj:
                block.mlp.forward = types.MethodType(forward_mlp_merged, block.mlp)
            else:
                block.mlp.forward = types.MethodType(forward_mlp, block.mlp)

            def forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                position_embeddings,
                **kwargs,
            ) -> torch.Tensor:
                residual = hidden_states
                hidden_states, norm_scaler = self.input_layernorm(hidden_states, return_scaler=True)
                # Self Attention
                hidden_states, _ = self.self_attn(
                    residual=residual,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                # Residuals are added in the output projection of the self attn block, instead of hidden_states = residual + hidden_states

                # Scaling is performed right before the silu after the projections
                residual = hidden_states
                hidden_states = self.mlp(hidden_states, residual, norm_scaler)
                return hidden_states

            block.forward = types.MethodType(forward, block)
