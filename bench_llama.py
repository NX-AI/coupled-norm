from transformers import pipeline, AutoModelForCausalLM
import time
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
        x = x.to(torch.float32)
        if scaler is None:
            var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
            scaler = torch.rsqrt(var)

        y = self.weight * (x * scaler).to(input_dtype)
        return (y, scaler) if return_scaler else y

def patch_ref_model(model):
    for block in model.model.layers:
        block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
        block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)


def patch_model(model):
    for block in model.model.layers:
        block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
        block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)

        def forward(
            self,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            cache_position,
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
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states, scaler=norm_scaler)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        block.forward = types.MethodType(forward, block) 

batch_size = 64
new_tokens = 128

def run_llama(pipe):
    return pipe(
        ["Your prompt here"] * batch_size, 
        max_new_tokens=new_tokens, 
        min_new_tokens=new_tokens,
        batch_size=batch_size
    )

def bench_llama(pipe):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.5)
    start = time.time()
    results = run_llama(pipe)
    torch.cuda.synchronize()
    end = time.time()
    time.sleep(0.5)
    return end - start

model_id = "meta-llama/Llama-3.2-3B"

model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=model_id, device="cuda")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
patch_model(model)
pipe.model = torch.compile(pipe.model)
warmup_results = run_llama(pipe)

exec_time_patched = bench_llama(pipe)
print(f"Time CoupledRMS: {exec_time_patched:.3f}\n")

del pipe
del model

model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
pipe = pipeline("text-generation", model=model, tokenizer=model_id, device="cuda")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
patch_ref_model(model)
pipe.model = torch.compile(pipe.model)
warmup_results = run_llama(pipe)

exec_time_ref = bench_llama(pipe)
print(f"Time REF: {exec_time_ref:.3f}\n")

speedup = exec_time_ref / exec_time_patched

print(f"REF/CoupledRMS: {exec_time_ref:.3f}/{exec_time_patched:.3f} speedup={speedup:.3f} model={model_id} batch_size={batch_size} new_tokens={new_tokens}")

# REF/CoupledRMS: 1.121/1.084 speedup=1.034 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=128
# REF/CoupledRMS: 4.501/4.363 speedup=1.032 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=512
# REF/CoupledRMS: 2.174/2.148 speedup=1.012 model=meta-llama/Llama-3.2-3B batch_size=128 new_tokens=128
# REF/CoupledRMS: 1.123/1.087 speedup=1.033 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=128
# REF/CoupledRMS: 1.767/1.712 speedup=1.032 model=meta-llama/Llama-3.2-3B batch_size=64 new_tokens=128
# REF/CoupledRMS: 18.427/17.909 speedup=1.029 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=2048
