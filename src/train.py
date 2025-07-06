#!/usr/bin/env python
"""
FP32 LoRA fine-tuning – avoids GradScaler issues.
"""

import os, json, pathlib, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH  = "data/cli_qa.jsonl"
OUT_DIR    = "lora_adapter"

BATCH = 4
GRAD_ACC = 8
LR = 2e-4
EPOCHS = 1
MAXLEN = 512

os.environ["ACCELERATE_DISABLE_TENSOR_PARALLEL"] = "1"

print("[1] Dataset")
rows = [json.loads(l) for l in pathlib.Path(DATA_PATH).open()]
raw  = Dataset.from_list(rows)

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tok.pad_token = tok.eos_token

def fmt(r):
    return tok(f"<s>[INST] {r['instruction']} [/INST] {r['response']} </s>",
               truncation=True, max_length=MAXLEN)
ds = raw.map(fmt, remove_columns=raw.column_names)

def collate(batch):
    pad = tok.pad(batch, return_tensors="pt")
    pad["labels"] = pad["input_ids"].clone()
    return pad

print("[2] Model & LoRA (FP32)")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
lora = get_peft_model(base, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], bias="none"
))
lora.enable_input_require_grads()  

print("[3] Train …")
steps = len(ds)//(BATCH*GRAD_ACC)+1
args  = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    max_steps=steps,
    fp16=False,                   
    logging_steps=10,
    save_total_limit=1,
    report_to="none",
)

trainer=Trainer(model=lora, args=args,train_dataset= ds, data_collator=collate,)
trainer.train()
print("[4] Save …")
lora.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(" adapter saved to", OUT_DIR)
