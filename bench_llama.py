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


def worker_bench(mode, model_id, batch_size, new_tokens, prompt, results_dict):
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
    pipe([prompt] * batch_size, max_new_tokens=new_tokens, min_new_tokens=new_tokens, batch_size=batch_size)
    
    torch.cuda.synchronize()
    start = time.time()
    _ = pipe([prompt] * batch_size, max_new_tokens=new_tokens, min_new_tokens=new_tokens, batch_size=batch_size)
    torch.cuda.synchronize()
    end = time.time()
    
    results_dict[mode] = end - start
    print(f"[{mode}] Finished. Time: {end-start:.3f}s")


model_id = "meta-llama/Llama-3.2-1B"
batch_size = 1
new_tokens = 256

prompt = """
A Turing machine is a mathematical model of computation describing an abstract machine[1] that manipulates symbols on a strip of tape according to a table of rules.[2] Despite the model's simplicity, it is capable of implementing any computer algorithm.[3]

The machine operates on an infinite[4] memory tape divided into discrete cells,[5] each of which can hold a single symbol drawn from a finite set of symbols called the alphabet of the machine. It has a "head" that, at any point in the machine's operation, is positioned over one of these cells, and a "state" selected from a finite set of states.
"""
prompt = "Who was Alan Turing?"


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    manager = mp.Manager()
    results = manager.dict()

    cooldown_dime = 10

    p1 = mp.Process(target=worker_bench, args=("REF", model_id, batch_size, new_tokens, prompt, results))
    p1.start()
    p1.join()
    print("Wait for GPU to cool down!")
    time.sleep(cooldown_dime)

    p2 = mp.Process(target=worker_bench, args=("COUPLED", model_id, batch_size, new_tokens, prompt, results))
    p2.start()
    p2.join()
    print("Wait for GPU to cool down!")
    time.sleep(cooldown_dime)

    p3 = mp.Process(target=worker_bench, args=("NONORM", model_id, batch_size, new_tokens, prompt, results))
    p3.start()
    p3.join()
    print("Wait for GPU to cool down!")
    time.sleep(cooldown_dime)

    p4 = mp.Process(target=worker_bench, args=("REFREF", model_id, batch_size, new_tokens, prompt, results))
    p4.start()
    p4.join()

    t_ref, t_coupled, t_no_norm, t_ref_ref = results["REF"], results["COUPLED"], results["NONORM"], results["REFREF"]
    refref = t_ref / t_ref_ref
    print("\nResults:")
    print(f"REF/CoupledRMS: {t_ref:.3f}/{t_coupled:.3f} speedup={t_ref / t_coupled:.3f} rp={1 - t_coupled / t_ref:.3f} model={model_id} batch_size={batch_size} new_tokens={new_tokens} refref={refref:.3f}")
    print(f"REF/NoNorm    : {t_ref:.3f}/{t_no_norm:.3f} speedup={t_ref / t_no_norm:.3f} rp={1 - t_no_norm / t_ref:.3f} model={model_id} batch_size={batch_size} new_tokens={new_tokens} refref={refref:.3f}")
    print(f"REFREF={refref:.3f}")

"""
Thermal Throttling changes the execution time, so we fix the clock rate to lessen that effect!
Fix clock speed (-i index of card):
nvidia-smi -lgc 1200 -i 0
Reset Clock Speed
nvidia-smi -rgc -i 0

# 3B
REF/CoupledRMS: 2.277/2.159 speedup=1.054 rp=0.052 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=256
REF/NoNorm    : 2.277/1.986 speedup=1.147 rp=0.128 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=256
REF/CoupledRMS: 2.700/2.550 speedup=1.059 rp=0.056 model=meta-llama/Llama-3.2-3B batch_size=16 new_tokens=256
REF/NoNorm    : 2.700/2.325 speedup=1.162 rp=0.139 model=meta-llama/Llama-3.2-3B batch_size=16 new_tokens=256
REF/CoupledRMS: 8.000/7.955 speedup=1.006 rp=0.006 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256
REF/NoNorm    : 8.000/7.562 speedup=1.058 rp=0.055 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256
REF/CoupledRMS: 5.603/5.299 speedup=1.057 rp=0.054 model=meta-llama/Llama-3.2-3B batch_size=16 new_tokens=512
REF/NoNorm    : 5.603/4.851 speedup=1.155 rp=0.134 model=meta-llama/Llama-3.2-3B batch_size=16 new_tokens=512

# 3B Train Forward
REF/CoupledRMS: 5.311/5.281 speedup=1.006 rp=0.006 model=meta-llama/Llama-3.2-3B batch_size=1024 new_tokens=1
REF/NoNorm    : 5.311/4.656 speedup=1.141 rp=0.123 model=meta-llama/Llama-3.2-3B batch_size=1024 new_tokens=1

# 1B
REF/CoupledRMS: 1.133/1.062 speedup=1.067 rp=0.063 model=meta-llama/Llama-3.2-1B batch_size=1 new_tokens=256
REF/NoNorm    : 1.133/0.965 speedup=1.175 rp=0.149 model=meta-llama/Llama-3.2-1B batch_size=1 new_tokens=256
REF/CoupledRMS: 1.374/1.296 speedup=1.060 rp=0.057 model=meta-llama/Llama-3.2-1B batch_size=16 new_tokens=256
REF/NoNorm    : 1.374/1.175 speedup=1.170 rp=0.145 model=meta-llama/Llama-3.2-1B batch_size=16 new_tokens=256
REF/CoupledRMS: 2.314/2.258 speedup=1.025 rp=0.024 model=meta-llama/Llama-3.2-1B batch_size=128 new_tokens=256
REF/NoNorm    : 2.314/2.116 speedup=1.094 rp=0.086 model=meta-llama/Llama-3.2-1B batch_size=128 new_tokens=256
REF/CoupledRMS: 4.126/4.083 speedup=1.010 rp=0.010 model=meta-llama/Llama-3.2-1B batch_size=256 new_tokens=256
REF/NoNorm    : 4.126/3.898 speedup=1.058 rp=0.055 model=meta-llama/Llama-3.2-1B batch_size=256 new_tokens=256
REF/CoupledRMS: 15.374/15.438 speedup=0.996 rp=-0.004 model=meta-llama/Llama-3.2-1B batch_size=1024 new_tokens=256
REF/NoNorm    : 15.374/15.023 speedup=1.023 rp=0.023 model=meta-llama/Llama-3.2-1B batch_size=1024 new_tokens=256

# 1B Train Forward
REF/CoupledRMS: 1.817/1.767 speedup=1.028 rp=0.027 model=meta-llama/Llama-3.2-1B batch_size=1024 new_tokens=1
REF/NoNorm    : 1.817/1.621 speedup=1.121 rp=0.108 model=meta-llama/Llama-3.2-1B batch_size=1024 new_tokens=1

# Fixed Termal
# 3B
REF/CoupledRMS: 12.391/12.043 speedup=1.029 rp=0.028 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256 refref=0.993 (REF, COUPLED, NO, REF)
REF/NoNorm    : 12.391/11.195 speedup=1.107 rp=0.097 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256 refref=0.993
REF/CoupledRMS: 12.489/12.133 speedup=1.029 rp=0.029 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256 refref=0.999 (COUPLED, REF, NO, REF)
REF/NoNorm    : 12.489/11.272 speedup=1.108 rp=0.097 model=meta-llama/Llama-3.2-3B batch_size=256 new_tokens=256 refref=0.999
REF/CoupledRMS: 3.689/3.423 speedup=1.078 rp=0.072 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=256 refref=0.991
REF/NoNorm    : 3.689/3.109 speedup=1.187 rp=0.157 model=meta-llama/Llama-3.2-3B batch_size=1 new_tokens=256 refref=0.991

# 1B
REF/CoupledRMS: 1.825/1.700 speedup=1.073 rp=0.068 model=meta-llama/Llama-3.2-1B batch_size=1 new_tokens=256 refref=1.002
REF/NoNorm    : 1.825/1.535 speedup=1.189 rp=0.159 model=meta-llama/Llama-3.2-1B batch_size=1 new_tokens=256 refref=1.002
"""
