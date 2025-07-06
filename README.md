# TinyLlama + Mini‑LoRA + Agent Demo

# TinyLlama-LoRA CLI Copilot

A one-file command-line copilot built on a 1.1 B-parameter TinyLlama model fine-tuned with LoRA adapters.  
It takes a plain-English task such as:

$ python src/agent.py "Create a new Git branch and switch to it"

and replies with...


Check current branch

Create and switch to the new branch

$ git branch
$ git checkout -b my-new-branch


The script also “dry-runs” every `$` command (echo-only) and stores a JSONL trace for later evaluation.

---

# Key points

 Model- [TinyLlama-1.1B-Chat v1.0](https://huggingface.co/TinyLlama) + LoRA adapter (4-bit, rank = 8) 
 Task - Convert natural-language CLI requests into numbered plans + POSIX shell commands 
 Trainin - 160 human-written NL → Shell pairs  
     1 epoch, 5 minutes on Colab T4  
     PEFT + QLoRA 
 Evaluation 
     evaluate_static.py  - (Rouge-L) 
     evaluate_dynamic.py - (manual 0-1-2 rubric) 
 Outputs 
     logs/trace.jsonl - (timestamped runs)  
     eval_static.md
     eval_dynamic.md
     report.md 
 Footprint
    2.2 GB base weights (HF cache) + 18 MB LoRA -> fits free Colab 

---

# Install & run

%%bash
git clone https://github.com/yourname/TinyLlama_LoRA_CLI_Copilot.git
cd TinyLlama_LoRA_CLI_Copilot

# 1. Python deps (CUDA 11.8 wheels pinned in requirements.txt)
pip install -r requirements.txt

# 2. Reproduce the adapter (optional – takes ~5 min)
python src/train.py

# 3. Quick test
python src/agent.py "List .py files recursively (dry-run)"



## Quick‑start (Colab / local GPU)
%%bash
cd /<my_directory>
pip install -r requirements.txt
python src/train.py                     
python src/evaluate_static.py           
awk '{print NR",https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot"$0}' eval_static.md    
python src/agent.py "List all .py files"
