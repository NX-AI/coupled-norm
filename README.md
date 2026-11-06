# coupled-rms-norm

[![Paper](https://img.shields.io/static/v1?label=Paper&message=OpenReview&color=8C1B13)](https://openreview.net/forum?id=DnPhmFIanE)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

This repository contains the code for CoupledNorm, introduced in the paper [CoupledNorm: Efficient Normalization via Shared RMS Statistics](https://openreview.net/forum?id=DnPhmFIanE), presented at the ICML 2026 Workshop AdaptFM.

Dependencies:
`pip install torch transformers hydra-core pandas`

## Benchmarking

Thermal Throttling changes the execution time, so we fix the clock rate to lessen that effect!
Fix clock speed (-i index of card):
nvidia-smi -lgc 1800 -i 0
Reset Clock Speed
nvidia-smi -rgc -i 0

Start Sweep:
`CUDA_VISIBLE_DEVICES=1 nohup python bench_model.py --config-name decode-sweep -m +cooldo
`n_time=40 >> bench-decode.out &`

## Why this archtecutre:

- Less components in the critical pyth
- Small Models: layernorm take quite a lot of time (O(n), but n isn't so big that it's only dominated by BMM's)
- Big Models: one less syncronisation barrier per block

Papers:
- Transformers without Normalization: https://arxiv.org/pdf/2503.10622
- Stronger Normalization-Free Transformers: https://arxiv.org/pdf/2512.10938
- SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm: https://arxiv.org/abs/2602.08064

## Cite

If you use CoupledNorm in your research, please cite our work:

```bibtex
@inproceedings{loretz2026couplednorm,
      title={CoupledNorm: Efficient Normalization via Shared {RMS} Statistics},
      author={Martin Loretz and Sepp Hochreiter},
      booktitle={ICML 2026 Workshop on Resource-Adaptive Foundation Model Inference (AdaptFM)},
      year={2026},
      url={https://openreview.net/forum?id=DnPhmFIanE},
}
```

## License

coupled-rms-norm is licensed under the [Apache License 2.0](./LICENSE).
