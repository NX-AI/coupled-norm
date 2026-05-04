import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rms_norm import patch_coupled_model_hf


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def generate_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True)

    if cfg.patch_model:
        patch_coupled_model_hf(model, merged=cfg.merged)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    pipe_out = pipe(cfg.prompt, do_sample=cfg.do_sample)
    generated_text = pipe_out[0]["generated_text"]
    print(generated_text)

if __name__ == "__main__":
    generate_app()


"""
How big is the universe? Is it possible to have a universe that is not a sphere? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between
"""
