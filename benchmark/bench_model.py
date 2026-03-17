import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import pipeline, AutoModelForCausalLM
import time
import torch
import torch.nn as nn
import types
import torch.multiprocessing as mp

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, weight=None, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.ones(hidden_size,))
        self.eps = eps

    def forward(self, x, scaler=None, return_scaler=False):
        if scaler is None:
            input_dtype = x.dtype
            var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps
            scaler = torch.rsqrt(var).to(input_dtype)

        y = self.weight * x * scaler
        return (y, scaler) if return_scaler else y

def patch_ref_model(model):
    for block in model.model.layers:
        block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
        block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)

def patch_no_norm_model(model):
    for block in model.model.layers:
        # For numerical stability (avoiding nan's) we scale them down even when using no norm!
        block.input_layernorm = Lambda(lambda x: x * 0.001)
        block.post_attention_layernorm = Lambda(lambda x: x * 0.001)

def patch_coupled_model(model):
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

def worker_bench(mode, model_id, batch_size, new_tokens, prompt, results_dict, warmup_iter):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda")
    pipe = pipeline("text-generation", model=model, tokenizer=model_id, device=0)
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    
    if mode == "REF":
        patch_ref_model(model)
    elif mode == "COUPLED":
        patch_coupled_model(model)
    elif mode == "NONORM":
        patch_no_norm_model(model)
    elif mode == "REFREF":
        patch_ref_model(model)

    pipe.model = torch.compile(pipe.model)
    for i in range(warmup_iter):
        pipe([prompt] * batch_size, max_new_tokens=new_tokens, min_new_tokens=new_tokens, batch_size=batch_size)

    torch.cuda.synchronize()
    start = time.time()
    _ = pipe([prompt] * batch_size, max_new_tokens=new_tokens, min_new_tokens=new_tokens, batch_size=batch_size)
    torch.cuda.synchronize()
    end = time.time()
    
    results_dict[mode] = end - start
    print(f"[{mode}] Finished. Time: {end-start:.3f}s")

def benchmark_model(model_id, prompt, batch_size, new_tokens, cooldown_time=20, warmup_iter=4):
    mp.set_start_method('spawn', force=True)

    manager = mp.Manager()
    results = manager.dict() 
    results["model_id"] = model_id
    results["prompt_len"] = len(prompt)
    results["batch_size"] = batch_size
    results["new_tokens"] = new_tokens

    variants = ["REF", "COUPLED", "NONORM", "REFREF"]
    for variant in variants:
        print(f"Starting {variant}")
        p = mp.Process(target=worker_bench, args=(variant, model_id, batch_size, new_tokens, prompt, results, warmup_iter))
        p.start()
        p.join()
        print("Wait for GPU to cool down!")
        time.sleep(cooldown_time)

    t_ref, t_coupled, t_no_norm, t_ref_ref = results["REF"], results["COUPLED"], results["NONORM"], results["REFREF"]
    refref = t_ref / t_ref_ref
    print("\nResults:")
    print(f"REF/CoupledRMS: {t_ref:.3f}/{t_coupled:.3f} speedup={t_ref / t_coupled:.3f} rp={1 - t_coupled / t_ref:.3f} model={model_id} batch_size={batch_size} new_tokens={new_tokens} refref={refref:.3f}")
    print(f"REF/NoNorm    : {t_ref:.3f}/{t_no_norm:.3f} speedup={t_ref / t_no_norm:.3f} rp={1 - t_no_norm / t_ref:.3f} model={model_id} batch_size={batch_size} new_tokens={new_tokens} refref={refref:.3f}")
    print(f"REFREF={refref:.3f}")

    return dict(results)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def benchmark_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    cooldown = cfg.cooldown_time if "cooldown_time" in cfg else 20
    warmup_iter = cfg.warmup_iter if "warmup_iter" in cfg else 4 
    results = benchmark_model(cfg["model_id"], cfg["prompt"], cfg["batch_size"], cfg["new_tokens"], cooldown_time=cooldown, warmup_iter=warmup_iter)

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    result_file = out_dir / "result.json"

    print(f"Saving results to {result_file}")

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # benchmark_model("meta-llama/Llama-3.2-1B", prompt="Who was Alan Turing?", batch_size=64, new_tokens=256) 
    benchmark_app()
