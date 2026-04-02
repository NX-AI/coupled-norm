#!/bin/bash
#SBATCH --output=out/out_%j.out
#SBATCH --reservation=nxai-timeseries
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20

source /home/martinloretz/space/coupled-rms-norm/nanogpt/.venv/bin/activate

echo "python train_gpt2.py $@"
python -u train_gpt2.py "$@"