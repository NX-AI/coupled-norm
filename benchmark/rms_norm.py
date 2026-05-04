import torch
import torch.nn as nn
import types

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
    coupled_downscale = config.coupled_downscale
    no_coupled_before = config.no_coupled_before if config.no_coupled_before is not None else -1
    no_coupled_after = config.no_coupled_after if config.no_coupled_after is not None else 100
    return coupled_downscale, no_coupled_before, no_coupled_after


def patch_coupled_model_hf(model, merged=False):
    coupled_downscale, no_coupled_before, no_coupled_after = coupled_config_from_hf(model)

    if merged:
        patch_coupled_qwen_merged(model, coupled_downscale, no_coupled_before, no_coupled_after)
    else:
        patch_coupled_model_manual(model, coupled_downscale, no_coupled_before, no_coupled_after)


"""
This fuction merges all operations from the second apply normalization step into other components, in a way that no kernel for normalization will be generated:
- Adddition of the residuals are done in the output projection of the attention layer
- The affinewise weights of the layernorm are statically merged into the up projection weights of the MLP
- The scaling of the normalization is merged into the silu kernel after the up projection, as that is mathematically equivilant to applying it before.
"""
def patch_coupled_qwen_merged(model, coupled_downscale, no_coupled_before, no_coupled_after):
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
            block.mlp.gate_proj.weight.data = (block.mlp.gate_proj.weight.data * ln_affine_weight_to_merge).clone()
            block.mlp.up_proj.weight.data = (block.mlp.up_proj.weight.data * ln_affine_weight_to_merge).clone()

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
                # attn_output = self.o_proj(attn_output)
                # Modified:
                # print(residual.shape, residual.dtype, attn_output.shape, attn_output.dtype, self.o_proj.weight.t().shape, self.o_proj.weight.t().dtype)
                out_shape = residual.shape
                attn_output = torch.addmm(residual.flatten(0, 1), attn_output.flatten(0, 1), self.o_proj.weight.t()).view(out_shape)

                return attn_output, attn_weights

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
                # attn_output = self.o_proj(attn_output)
                # Modified:
                # print(residual.shape, residual.dtype, attn_output.shape, attn_output.dtype, self.o_proj.weight.t().shape, self.o_proj.weight.t().dtype)
                out_shape = residual.shape
                attn_output = torch.addmm(residual.flatten(0, 1), attn_output.flatten(0, 1), self.o_proj.weight.t()).view(out_shape)

                return attn_output, attn_weights


            def forward_mlp(self, x, scaler):
                gate = (scaler * self.gate_proj(x)).to(x.dtype)
                up = (scaler * self.up_proj(x)).to(x.dtype)
                return self.down_proj(self.act_fn(gate) * up)


            model_type = model.config.model_type
            if model_type == "qwen3":
                block.self_attn.forward = types.MethodType(forward_attn_qwen, block.self_attn)
            elif model_type == "llama":
                block.self_attn.forward = types.MethodType(forward_attn_llama, block.self_attn)
            else:
                raise Exception(f"Merging isn't supported for {model_type}")

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

                residual = hidden_states
                # Scaling is performed right before the silu after the projections
                hidden_states = self.mlp(hidden_states, norm_scaler)
                hidden_states = residual + hidden_states
                return hidden_states

            block.forward = types.MethodType(forward, block)
