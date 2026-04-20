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

source /home/martinloretz/space/coupled-rms-norm/distill/.venv/bin/activate

echo "python components/knowledge_distillation_single_device.py $@"
python components/knowledge_distillation_single_device.py $@
