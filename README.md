# coupled-rms-norm

Dependencies:
`pip install torch transformers hydra-core`

## Benchmarking

Thermal Throttling changes the execution time, so we fix the clock rate to lessen that effect!
Fix clock speed (-i index of card):
nvidia-smi -lgc 1200 -i 0
Reset Clock Speed
nvidia-smi -rgc -i 0
