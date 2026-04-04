# Distillation

## Install
`pip install torchtune torch torchvision torchao==0.7`
`pip install "lm_eval[hf]==0.4.7" transformers==4.48.3`
`pip install torch==2.10 torchvision --index-url https://download.pytorch.org/whl/cu126`

## Run

### Download Model:
`tune download meta-llama/Llama-3.2-1B-Instruct --output-dir model/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <TOKEN>`

### Distill CoupledRMS
`tune run knowledge_distillation_single_device --config config/llama/distill_llama_1B.yaml model._component_=components.llama3_2_1B_coupled_rms.lora_llama3_2_1b model.norm_normal_before=3 output_dir=output/distill/coupled/test/`

### Distill Baseline
`tune run knowledge_distillation_single_device --config config/llama/distill_llama_1B.yaml model._component_=components.llama3_2_1B_coupled_rms.lora_llama3_2_1b model.norm_normal_before=100 output_dir=output/distill/baseline/test/`

### Distill Baseline to CoupledRMS
`tune run knowledge_distillation_single_device --config config/llama/distill_llama_1B.yaml  model._component_=components.llama3_2_1B_coupled_rms.lora_llama3_2_1b model.norm_normal_before=3 teacher_model._component_=components.llama3_2_1B_coupled_rms.llama3_2_1b teacher_model.norm_normal_before=100 teacher_checkpointer.checkpoint_dir=output/distill/baseline/test/epoch_0  teacher_checkpointer.checkpoint_files=[model-00001-of-00001.safetensors] output_dir=output/distill/coupled-from-baseline/test/`

### Eval CoupledRMS
`tune run eleuther_eval --config config/llama/eval_llama_1b.yaml model._component_=components.llama3_2_1B_coupled_rms.llama3_2_1b model.norm_normal_before=3 checkpointer.checkpoint_dir=output/distill/coupled/test/epoch_0`

### Eval Baseline
`tune run eleuther_eval --config config/llama/eval_llama_1b.yaml model._component_=components.llama3_2_1B_coupled_rms.llama3_2_1b model.norm_normal_before=100 checkpointer.checkpoint_dir=output/distill/baseline/test/epoch_0`

### Eval Base Checkpoint
`tune run eleuther_eval --config config/llama/eval_llama_1b.yaml model._component_=torchtune.models.llama3_2.llama3_2_1b checkpointer.checkpoint_files=[model.safetensors] checkpointer.checkpoint_dir=model/Llama-3.2-1B-Instruct`
