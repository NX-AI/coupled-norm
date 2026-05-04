import json
import time
import gc
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import pipeline
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity
from rms_norm import patch_coupled_model_manual

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

def patch_ref_model(model):
    for block in model.model.layers:
        block.input_layernorm = RMSNorm(model.config.hidden_size, weight=block.input_layernorm.weight.data, eps=model.config.rms_norm_eps)
        block.post_attention_layernorm = RMSNorm(model.config.hidden_size, weight=block.post_attention_layernorm.weight.data, eps=model.config.rms_norm_eps)

def patch_no_norm_model(model):
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    for block in model.model.layers:
        # For numerical stability (avoiding nan's) we scale them down even when using no norm!
        block.input_layernorm = Lambda(lambda x: x * 0.001)
        block.post_attention_layernorm = Lambda(lambda x: x * 0.001)


def performance_benchmark_model(cfg: DictConfig):
    pipe = pipeline("text-generation", model=cfg.model_id, device="cuda:0", torch_dtype=torch.float16, trust_remote_code=True)
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    num_threads = torch.get_num_threads()

    if cfg.mode == "ref":
        patch_ref_model(pipe.model)
    elif cfg.mode == "coupled":
        if "qwen" in cfg.model_id.lower():
            coupled_downscale = 0.92
            norm_normal_before = 0
        elif "llama" in cfg.model_id.lower():
            coupled_downscale = 0.91743
            norm_normal_before = 3
        else:
            raise Exception("No known model configuration!")
        patch_coupled_model_manual(pipe.model, coupled_downscale, norm_normal_before, no_coupled_after=1000)
    elif cfg.mode == "nonorm":
        patch_no_norm_model(pipe.model)
    else:
        raise Exception(f"No known mode {cfg.mode}")

    pipe.model = torch.compile(pipe.model, mode="reduce-overhead")

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    for i in range(cfg.warmup):
        pipe(prompts, max_new_tokens=cfg.new_token_count, min_new_tokens=cfg.new_token_count, batch_size=cfg.batch_size)


    timer = benchmark.Timer(
        stmt='pipe(prompts, max_new_tokens=new_token_count, min_new_tokens=new_token_count, batch_size=batch_size)',
        globals={'pipe': pipe, 'prompts': prompts, 'new_token_count': cfg.new_token_count, 'batch_size': cfg.batch_size},
        num_threads=num_threads
    )

    measurement = timer.blocked_autorange(min_run_time=cfg.autorange_min_run_time)
    return measurement, {}



def benchmark_model(cfg : DictConfig):
    measurement_to_dict = lambda measure: {"mean": measure.mean, "median": measure.median, "p25": measure._p25, "p75": measure._p75, "iqr": measure.iqr, "num_threads": measure.task_spec.num_threads, "times": measure._sorted_times}

    measurement, extra_keys = performance_benchmark_model(cfg)
    print(measurement)

    return {
        **dict(cfg),
        **extra_keys,
        **measurement_to_dict(measurement)
    }

@hydra.main(version_base=None, config_path="config", config_name="config")
def benchmark_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if (cfg.profile if "profile" in cfg else False):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            results = benchmark_model(cfg)
        prof.export_chrome_trace("trace.json")
    else:
        results = benchmark_model(cfg)

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    result_file = out_dir / "result.json"

    print(f"Saving results to {result_file}")

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    gc.collect()
    if "sleep" in cfg:
        time.sleep(cfg.sleep)

if __name__ == "__main__":
    benchmark_app()
