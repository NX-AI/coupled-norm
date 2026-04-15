#!/bin/bash
#SBATCH --output=out/out_%j.out
#SBATCH --reservation=nxai-timeseries
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=cancel

set -euo pipefail

name=$1
shift

source /home/martinloretz/space/coupled-rms-norm/distill/.venv/bin/activate

echo "tune run knowledge_distillation_single_device --config config/llama/distill_llama_1B_nemotron.yaml  output_dir=/nfs-gpu/users_home/martin/couplednorm-output/distill/coupled-${name}/ $@"
tune run knowledge_distillation_single_device --config config/llama/distill_llama_1B_nemotron.yaml  output_dir=/nfs-gpu/users_home/martin/couplednorm-output/distill/coupled-${name}/ $@

echo "tune run eleuther_eval --config config/llama/eval_llama_1b.yaml checkpointer.checkpoint_dir=/nfs-gpu/users_home/martin/couplednorm-output/distill/coupled-${name}/epoch_0"
tune run eleuther_eval --config config/llama/eval_llama_1b.yaml checkpointer.checkpoint_dir=/nfs-gpu/users_home/martin/couplednorm-output/distill/coupled-${name}/epoch_0

echo "coupled-${name}"
