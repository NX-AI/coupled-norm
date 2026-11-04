# pip install torch --index-url https://download.pytorch.org/whl/cu126
# pip install "lm-eval[hf]" transformers datasets sentencepiece protobuf hydra-core

import json
from pathlib import Path

import hydra
import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from omegaconf import DictConfig, OmegaConf
from rms_norm import patch_coupled_model_hf
from transformers import AutoModelForCausalLM, AutoTokenizer


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def eval_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True
    )

    if cfg.patch_model:
        patch_coupled_model_hf(model)

    model = torch.compile(model)

    output = lm_eval.simple_evaluate(
        model=HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size="auto", device="cuda"
        ),
        tasks=list(cfg.tasks),
        num_fewshot=cfg.shots,
    )

    print(make_table(output))

    results = output["results"]

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    result_file = out_dir / "result.json"

    print(f"Saving results to {result_file}")

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    eval_app()
