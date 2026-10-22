"""
mac_pipeline/lora/train_lora.py
-------------------------------
Train one LoRA adapter that teaches a base model one subject, log the training energy with
CodeCarbon, and push the adapter (30-40 MB) to a private Hugging Face repo.

Runs on Colab (T4) or locally. Plain `transformers.Trainer` with a hand-written collator so it
does not depend on TRL's changing API. Loss is computed on the assistant's tokens only.

  python train_lora.py --base Qwen/Qwen2.5-3B-Instruct --subject math \
      --push-to user/qwen2.5-3b-lora-math --n 2000 --epochs 2

Subjects
  math       GSM8K train, reason-then-answer format ending in "Answer: <number>"
  code       MBPP train (prompt + tests -> code)
  knowledge  MMLU auxiliary_train, our exact letter-scoring prompt ending in " <letter>"
"""

import argparse
import json
import os
import platform
import time

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

LETTERS = "ABCDE"
COT_INSTRUCTION = ("Think step by step, briefly. When you are done, finish with a line of the "
                   "form 'Answer: X' where X is the final answer.")


# ----------------------------------------------------------------------------- data
def format_prompt(question, choices):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


def build_examples(subject, n, seed=0):
    """Returns a list of {"prompt": str, "completion": str, "chat": bool}.
    chat=True -> wrapped in the chat template (user/assistant); chat=False -> raw continuation,
    matching how the Mac pipeline scores that subject."""
    ex = []
    if subject == "math":
        ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
        for r in ds:
            sol, ans = r["answer"].split("####")
            sol = sol.strip().replace("<<", "").replace(">>", "")   # drop calculator annotations
            import re
            sol = re.sub(r"\([\d\.\+\-\*/ ]+=[\d\.]+\)", "", sol)
            ex.append(dict(prompt=r["question"].strip() + "\n\n" + COT_INSTRUCTION,
                           completion=sol + f"\nAnswer: {ans.strip()}", chat=True))
    elif subject == "code":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train").shuffle(seed=seed)
        ds = ds.select(range(min(n, len(ds))))
        for r in ds:
            tests = "\n".join(r["test_list"][:2])
            ex.append(dict(prompt=f"{r['text'].strip()}\nYour code should pass these tests:\n{tests}\n"
                                  "Reply with the Python code only.",
                           completion="```python\n" + r["code"].strip() + "\n```", chat=True))
    elif subject == "knowledge":
        ds = load_dataset("cais/mmlu", "auxiliary_train", split="train").shuffle(seed=seed).select(range(n))
        for r in ds:
            ex.append(dict(prompt=format_prompt(r["question"].strip(), [str(c).strip() for c in r["choices"]]),
                           completion=f" {LETTERS[int(r['answer'])]}", chat=False))
    else:
        raise ValueError(subject)
    return ex


class SFTDataset(torch.utils.data.Dataset):
    """Tokenises prompt+completion; labels are -100 on the prompt so loss covers the answer only."""

    def __init__(self, examples, tok, max_len):
        self.items = []
        for e in examples:
            if e["chat"]:
                p_ids = tok.apply_chat_template([{"role": "user", "content": e["prompt"]}],
                                                add_generation_prompt=True, tokenize=True)
                if hasattr(p_ids, "keys"):
                    p_ids = p_ids["input_ids"]
                c_ids = tok(e["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            else:
                p_ids = tok(e["prompt"], add_special_tokens=True)["input_ids"]
                c_ids = tok(e["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (list(p_ids) + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); lab[i, :n] = torch.tensor(b["labels"]); att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subject", required=True, choices=["math", "code", "knowledge"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    ap.add_argument("--four-bit", action="store_true", help="QLoRA (CUDA + bitsandbytes only)")
    ap.add_argument("--out", default="adapter_out")
    ap.add_argument("--push-to", default=None, help="HF repo id, e.g. user/qwen2.5-3b-lora-math")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--country", default="IND")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {dev}  base {args.base}  subject {args.subject}  n {args.n}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if args.four_bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model = AutoModelForCausalLM.from_pretrained(
            args.base, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    examples = build_examples(args.subject, args.n, args.seed)
    ds = SFTDataset(examples, tok, args.max_len)
    print(f"{len(ds)} examples, mean tokens {sum(len(x[0]) for x in ds.items) / len(ds):.0f}", flush=True)

    steps_per_epoch = max(1, len(ds) // (args.batch * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup = max(1, int(0.03 * total_steps))        # warmup_ratio was removed in transformers 5
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup,
        logging_steps=10, save_steps=200, save_total_limit=2, report_to=[],
        fp16=(dev == "cuda"), bf16=False, optim="adamw_torch", seed=args.seed,
        dataloader_pin_memory=(dev == "cuda"), remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))

    # ---- energy of the training run
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code=args.country, measure_power_secs=5,
                                          output_dir=args.out, log_level="error", save_to_file=True)
        tracker.start()
    except Exception as e:
        print("codecarbon unavailable:", e)
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    energy = {}
    if tracker:
        tracker.stop()
        d = tracker.final_emissions_data
        energy = dict(kwh=d.energy_consumed, gco2e=d.emissions * 1000, gpu_kwh=d.gpu_energy,
                      cpu_kwh=d.cpu_energy, duration_s=d.duration, country=d.country_iso_code)

    # ---- save adapter + summary
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    import transformers, peft
    summary = dict(base=args.base, subject=args.subject, n_examples=len(ds), epochs=args.epochs,
                   steps=int(result.global_step), train_loss=float(result.training_loss),
                   rank=args.rank, alpha=args.alpha, lr=args.lr, max_len=args.max_len,
                   four_bit=args.four_bit, device=dev, gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else dev),
                   wall_s=wall, energy=energy, transformers=transformers.__version__, peft=peft.__version__,
                   torch=torch.__version__, python=platform.python_version())
    with open(os.path.join(adapter_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    if args.push_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to, private=True, exist_ok=True)
        api.upload_folder(folder_path=adapter_dir, repo_id=args.push_to)
        print(f"pushed adapter to https://huggingface.co/{args.push_to}", flush=True)


if __name__ == "__main__":
    main()
