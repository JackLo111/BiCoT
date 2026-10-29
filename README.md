# BiCoT

Code for the core experiment of *Echoes within the Reasoning: Stealthy and Effective Watermarking via
Chain of Thought* (ICML 2026): inject a watermark into the reasoning-trace geometry of an LLM, then
verify it from sentinel-token logits only (RSR).

## What is included
| file | role |
|---|---|
| `bicot/watermark.py` | secret vector `v` and the inner-loop injection `delta* = alpha * v` |
| `bicot/trainer.py` | bi-level objective: task CE under injection + anchor-collapse + control-orthogonality + KL |
| `bicot/data.py` | GSM8K CoT data; anchors = numeric tokens in the reasoning part |
| `bicot/rsr_verifier.py` | Robust Subspace Registration (sentinels, median/MAD drift calibration, correlation score) |
| `bicot/train.py` | LoRA watermark injection (`bicot-train`) |
| `bicot/detect.py` | black-box verification: watermarked model vs. its base model (`bicot-detect`) |

## Setup
Requires a CUDA GPU (~30 GB for the 7B model) and [uv](https://docs.astral.sh/uv/).
```bash
uv sync                      # creates .venv with the pinned, tested dependencies
```

## Run
```bash
uv run bicot-train  --output_dir ./outputs/bicot_qwen7b      # ~25 min on one A100
uv run bicot-detect --adapter_dir ./outputs/bicot_qwen7b
```
Quick smoke test of training: add `--max_steps 5`. GSM8K and the base model are downloaded from the
Hugging Face Hub on first use. Development: `uv sync --group dev && uv run ruff check .`

`bicot-detect` reports AUC, pAUC (FPR <= 0.05), Mahalanobis distance (MD) and TPR at fixed FPRs.

## Citation
If you use this code, please cite:
```bibtex
@inproceedings{pmlr-v306-lu26q,
 title={Echoes within the Reasoning: Stealthy and Effective Watermarking via Chain of Thought},
 author={Lu, Jiacheng and Li, Yiming and Song, Tao and Wang, Weijian and Qu, Wenjie and Guan, Haibing and Zhang, Jiaheng},
 booktitle={Proceedings of the 43rd International Conference on Machine Learning},
 series={Proceedings of Machine Learning Research}, volume={306}, pages={81212--81235}, year={2026},
 publisher={PMLR}, url={https://proceedings.mlr.press/v306/lu26q.html}
}
```
