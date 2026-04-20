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


def patch_coupled_model_hf(model):
    config = model.config
    coupled_downscale = config.coupled_downscale
    no_coupled_before = config.no_coupled_before if config.no_coupled_before is not None else -1
    no_coupled_after = config.no_coupled_after if config.no_coupled_after is not None else 100

    patch_coupled_model_manual(model, coupled_downscale, no_coupled_before, no_coupled_after)
