<!-- README.md — Attention-Fusion DeepFake Detector (example)  -->
<p align="center">
  <!-- Optional banner / logo -->
  <!-- <img src="assets/banner.png" alt="Attention-Fusion DeepFake Detector" width="60%"> -->
</p>

<div align="center">

<!-- Badges: feel free to remove or swap -->
<a href="https://github.com/snehakumari1996/Attention_Network_for_Deepfake_Detection/actions">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/snehakumari1996/Attention_Network_for_Deepfake_Detection/ci.yml?logo=github">
</a>
<a href="https://github.com/snehakumari1996/Attention_Network_for_Deepfake_Detection/stargazers">
  <img alt="Stars" src="https://img.shields.io/github/stars/snehakumari1996/Attention_Network_for_Deepfake_Detection?color=yellow&logo=star">
</a>
<a href="LICENSE">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green">
</a>
<a href="docs/paper.pdf">
  <img alt="Paper" src="https://img.shields.io/badge/paper-ICAAIML 2024-orange">
</a>

</div>

<br/>

## 🧩 What’s inside?
| Metric / Feature | Value |
|------------------|-------|
| CelebDF-v2 **AUC-ROC** | **0.998** |
| Accuracy | 98.1 % |
| Architecture | Encoder-Decoder ＋ Attention Fusion |
| Framework | PyTorch 1.13 |
| Training Time | ~4 h on single A100 |
| License | MIT |

> **TL;DR** — This repo contains the code, weights, and training scripts for an attention-fusion network that detects DeepFakes in RGB **and** frequency domains, matching state-of-the-art results with <10 k trainable parameters.

---

## 🚀 Quick start

```bash
# 1. Create environment
conda create -n deepfake python=3.10
conda activate deepfake
pip install -r requirements.txt

# 2. Download pretrained weights
python scripts/download_weights.py   # 60 MB

# 3. Run inference on a folder of videos
python infer.py --input samples/ --output results/
