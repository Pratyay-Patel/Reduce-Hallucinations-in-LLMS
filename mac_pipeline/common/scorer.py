"""
mac_pipeline/common/scorer.py
-----------------------------
Option-likelihood scorer, ported from poc/score_options.py for Apple Silicon (MPS) with
fp16 weights, 3-5 answer options, and no quantisation paths.

For each question: one forward pass per option letter, mean token log-prob of the letter
given the prompt, argmax = prediction, softmax over the options = graded confidence.
Zero generated tokens, so no parsing failures and ~0.3-1 s per question.
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import LETTERS, format_prompt


class OptionScorer:
    def __init__(self, model_id: str, device: str, chat_template: bool = False,
                 max_length: int = 1024, adapter: str | None = None):
        # 1024 > the longest prompt (641 Qwen tokens for MMLU) plus any chat wrapper, so the
        # left-truncation in _option_logprob never fires on this question bank
        self.model_id = model_id
        self.device = device
        self.chat_template = chat_template
        self.max_length = max_length
        self.adapter = adapter
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        if adapter:
            # a LoRA adapter (HF repo id or local dir) trained on exactly this base; kept
            # unmerged so several adapters can share one loaded base (load_adapter/set_adapter)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model = self.model.to(device).eval()
        self.reset_peak()

    def add_adapter(self, name: str, adapter: str):
        """Attach another adapter to the same base; switch with `use_adapter(name)`."""
        self.model.load_adapter(adapter, adapter_name=name)

    def use_adapter(self, name: str):
        self.model.set_adapter(name)

    # ------------------------------------------------------------------ memory helpers
    def reset_peak(self):
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def peak_mem_gb(self) -> float:
        if self.device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        if self.device.startswith("cuda"):
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0

    def unload(self):
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ prompt encoding
    @staticmethod
    def build_prompt_ids(tok, prompt: str, chat_template: bool):
        """Returns (input_ids [1,T] on CPU, letter_formatter). Model-free, so testable."""
        if chat_template:
            msgs = [{"role": "user", "content": prompt + "\nReply with the letter only."}]
            # enable_thinking=False: Qwen3 templates otherwise open a <think> block in the
            # assistant turn; other templates ignore the variable
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                          return_tensors="pt", return_dict=True)
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            return ids, (lambda i: LETTERS[i])            # assistant turn starts fresh: no space
        ids = tok(prompt, return_tensors="pt").input_ids
        return ids, (lambda i: f" {LETTERS[i]}")          # continuation after 'Answer:'

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def _option_logprob(self, prompt_ids: torch.Tensor, option_text: str) -> float:
        opt_ids = self.tok(option_text, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(prompt_ids.device)
        ids = torch.cat([prompt_ids, opt_ids], dim=1)[:, -self.max_length:]
        logits = self.model(ids).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        n_opt = opt_ids.shape[1]
        tgt = ids[0, -n_opt:]
        tok_lp = logp[-n_opt:].gather(1, tgt[:, None]).squeeze(1)
        return tok_lp.mean().item()

    # ------------------------------------------------------------------ reason, then score letters
    COT_INSTRUCTION = ("Think step by step, briefly. When you are done, finish with a line of the "
                       "form 'Answer: X' where X is the letter of the correct option.")
    COT_STEM = "\nTherefore, the answer is ("

    @torch.inference_mode()
    def score_cot(self, question: str, choices: list, gold: int, max_new_tokens: int = 256) -> dict:
        """Let the model write its reasoning (greedy, bounded), then append a fixed answer stem
        and score the option letters exactly as `score` does, conditioned on the reasoning.
        The answer is never parsed from free text. Always uses the chat template: instruct
        models reason in that format; the letter is scored as the continuation of the stem."""
        import re
        t0 = time.perf_counter()
        prompt = format_prompt(question, choices).rsplit("\nAnswer:", 1)[0]
        msgs = [{"role": "user", "content": prompt + "\n\n" + self.COT_INSTRUCTION}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           return_tensors="pt", return_dict=True)
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(self.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        gen_ids = out[0, ids.shape[1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        # strip a trailing EOS so the stem continues the assistant turn
        gen_ids = gen_ids[gen_ids != self.tok.eos_token_id] if self.tok.eos_token_id is not None else gen_ids
        stem_ids = self.tok(self.COT_STEM, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        ctx = torch.cat([ids, gen_ids[None, :], stem_ids], dim=1)
        lps = [self._option_logprob(ctx, LETTERS[i]) for i in range(len(choices))]
        lps_t = torch.tensor(lps)
        probs = torch.softmax(lps_t, dim=0)
        pred = int(lps_t.argmax())
        m = re.findall(r"Answer:\s*\(?([A-E])\)?", text)
        stated = LETTERS.index(m[-1]) if m else None
        if self.device == "mps":
            torch.mps.synchronize()
        return dict(
            pred=pred, pred_letter=LETTERS[pred], correct=int(pred == gold),
            graded=float(probs[pred]), p_gold=float(probs[gold]),
            logprobs=[round(x, 5) for x in lps],
            prompt_tokens=int(ids.shape[1]), reasoning_tokens=int(gen_ids.shape[0]),
            truncated=int(gen_ids.shape[0] >= max_new_tokens),
            stated_letter=(LETTERS[stated] if stated is not None else None),
            stated_matches_scored=(None if stated is None else int(stated == pred)),
            reasoning=text,
            rendered_prompt=self.tok.decode(ids[0], skip_special_tokens=False),
            seconds=time.perf_counter() - t0,
            peak_mem_gb=self.peak_mem_gb(),
        )

    def score(self, question: str, choices: list, gold: int) -> dict:
        t0 = time.perf_counter()
        prompt = format_prompt(question, choices)
        prompt_ids, letter = self.build_prompt_ids(self.tok, prompt, self.chat_template)
        prompt_ids = prompt_ids.to(self.device)
        lps = [self._option_logprob(prompt_ids, letter(i)) for i in range(len(choices))]
        lps_t = torch.tensor(lps)
        probs = torch.softmax(lps_t, dim=0)
        pred = int(lps_t.argmax())
        if self.device == "mps":
            torch.mps.synchronize()
        return dict(
            pred=pred, pred_letter=LETTERS[pred], correct=int(pred == gold),
            graded=float(probs[pred]), p_gold=float(probs[gold]),
            logprobs=[round(x, 5) for x in lps],
            prompt_tokens=int(prompt_ids.shape[1]),
            # exact text the model saw (chat wrapper and special tokens included); the scored
            # option letter is appended after this
            rendered_prompt=self.tok.decode(prompt_ids[0], skip_special_tokens=False),
            seconds=time.perf_counter() - t0,
            peak_mem_gb=self.peak_mem_gb(),
        )
