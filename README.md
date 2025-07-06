# TinyLlama + Mini‑LoRA + Agent Demo

All paths assume you cloned or unzipped this repo to:
`/content/drive/MyDrive/fenrir-mini-lora-agent`

## Quick‑start (Colab / local GPU)
```bash
cd /content/drive/MyDrive/fenrir-mini-lora-agent
pip install -r requirements.txt
python src/train.py                      # ≈20 min on free T4
python src/evaluate_static.py            # writes eval_static.md
awk '{print NR",https://github.com/snehakumari1996/TinyLlama_LoRA_CLI_Copilot"$0}' eval_static.md    # preview table
python src/agent.py "List all .py files"
