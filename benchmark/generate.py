import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rms_norm import patch_coupled_model_hf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def generate_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )

    if cfg.patch_model:
        patch_coupled_model_hf(
            model, merged=cfg.merged, merge_up_proj=cfg.merge_up_proj
        )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    pipe_out = pipe(cfg.prompt, do_sample=cfg.do_sample)
    generated_text = pipe_out[0]["generated_text"]
    print(generated_text)


if __name__ == "__main__":
    generate_app()


# How big is the universe? Is it possible to have a universe that is not a sphere? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between a sphere and a non-spherical object? What is the difference between

# How big is the universe? How does it relate to the concept of a "cosmic" or "cosmic" space? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between the universe and the cosmos? What is the difference between

# Unmerged:
# How big is the universe? How does it relate to the concept of the universe as a single entity? What is the relationship between the universe and the concept of the universe as a single entity? What is the relationship between the universe and the concept of the universe as a single entity? What is the relationship between the universe and the concept of the universe as a single entity?
#
# Okay, so I need to figure out how big the universe is and how it relates to the idea of the universe as a single entity. Let me start by recalling what I know. The universe is vast, right? But how big exactly? I think it's about a trillion times the size of the Earth. Wait, is that accurate? I remember reading somewhere that the universe is about 13.8 billion light-years in diameter. But I'm not sure if that's correct. Maybe I should check my memory. Also, the concept of the universe as a single entity... Hmm, does that mean it's just one big entity, or is there more to it? Like, maybe different parts of the universe are separate? Or is it a single, unified whole?
#
# I know that the universe is made up of galaxies, stars, planets, and other celestial bodies. So, if the universe is a single entity,
