<!-- ──────────────────────────────────────────────────────────────────────────────
TinyLlama-LoRA CLI Copilot · README
A drop-in, copy-paste-ready README.md
────────────────────────────────────────────────────────────────────────────── -->

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" width="140">
  <br>
  <strong>TinyLlama + Mini-LoRA + Agent Demo</strong><br>
  <em>Turn plain-English CLI requests into numbered plans&nbsp;+ shell commands</em>
  <br><br>
  <a href="https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/snehakumari1996/TinyLlama_LoRA_CLI_Copilot"></a>
  <a href="https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/snehakumari1996/TinyLlama_LoRA_CLI_Copilot"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue">
</p>

---

##  What is this?

TinyLlama-LoRA CLI Copilot is a **one-file command-line assistant**:

<code>$ python src/agent.py "Create a new Git branch and switch to it"   </code>
1. Check current branch
2. Create and switch to the new branch

   <code>$ git branch</code><br>
   <code>$ git checkout -b my-new-branch</code>


It also *dry-runs* every `$` command (echo-only) and logs a JSONL trace for later
evaluation.

---

##   Key points

| Feature | Detail |
|---------|--------|
| **Model** | [TinyLlama-1.1B-Chat v1.0](https://huggingface.co/TinyLlama) + 18 MB LoRA adapter (rank = 8, 4-bit) |
| **Task** | Convert NL CLI queries ⟶ numbered plan + POSIX shell commands |
| **Training** | 160 human-written (NL → shell) pairs · QLoRA · 1 epoch (~5 min on Colab T4) |
| **Evaluation** | `evaluate_static.py` (ROUGE-L) · `evaluate_dynamic.py` (manual 0-1-2 rubric) |
| **Footprint** | 2.2 GB base weights (HF cache) + 18 MB adapter ➜ fits free Colab |
| **Zero-to-run** | `pip install -r requirements.txt` · `python src/agent.py "…"` |

---

## Quick-start
<code>
```bash
# clone + install
git clone https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot.git
cd TinyLlama_LoRA_CLI_Copilot
pip install -r requirements.txt          # CUDA 11.8 wheels pinned

# (optional) reproduce the adapter – ~5 min on Colab T4
python src/train.py

# run a demo
python src/agent.py "List .py files recursively (dry-run)"
</code>



Repo layout

.
├── lora_adapter/          
├── logs/
│   └── trace.jsonl       
├── offload/               
├── src/
│   ├── agent.py            
│   ├── evaluate_static.py  
│   └── evaluate_dynamic.py 
├── eval_static.md         
├── eval_dynamic.md         
├── requirements.txt        
└── report.md               





