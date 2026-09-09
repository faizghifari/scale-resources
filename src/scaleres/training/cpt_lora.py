#!/usr/bin/env python
"""Continued pretraining via LoRA, on a single consumer GPU, without Unsloth.

Written for the Cirebonese generator experiment: every generator tested writes
Javanese because none has real Cirebonese exposure (F61-F65), and no prompt
lever fixes that -- the best recipe SUPPRESSES Javanese (2.04 -> 0.42 marker
rate) but never gets the model to produce Cirebonese-exclusive vocabulary it
does not have. Putting the language into the weights is the only untried route.

Deliberately plain peft rather than the repo's Unsloth path: Unsloth pins torch
and transformers hard, this stack is pinned for other reasons, and P0.4 already
cost a day to unbreak. bf16 LoRA without 4-bit quantisation also avoids adding
bitsandbytes. On a 16 GB card that budget is roughly:

    weights   4.02B x 2 bytes            ~8.0 GB
    LoRA r32 over 7 projections           ~0.13 GB
    AdamW fp32 states for adapters only   ~0.52 GB
    activations, checkpointing, seq 1024  ~1-2 GB

which leaves headroom. Full fine-tuning would need ~48 GB and is not the point:
with under 2M tokens of target text, LoRA is also the correct capacity.

    python -m scaleres.training.cpt_lora \\
        --base models/base/Qwen3-4B-Instruct-2507 \\
        --train dataset/cpt/cbn_train_disjoint dataset/cpt/cbn_tierB \\
        --out models/Qwen3-4B-cbn-lora
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batch", type=int, default=2,
                    help="HF defaults this to 8, which on a 16GB card allocates "
                         "a block big enough to fragment the allocator for the "
                         "rest of the run -- steps went 19s -> 7.7min at 98% "
                         "VRAM, a thrash rather than an OOM")
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to resume from")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 QLoRA. Needed for anything above ~5B on a 16GB "
                         "card. Note it is a second variable: comparing a 4-bit "
                         "model against a bf16 one confounds quantisation with "
                         "whatever else differs, so run a matched control.")
    ap.add_argument("--bos-per-block", choices=("auto", "on", "off"),
                    default="auto",
                    help="Start every packed block with BOS. auto turns it on "
                         "when the tokenizer defines a BOS but does not add "
                         "one (Gemma). Without it Gemma scores worse than "
                         "chance on raw text.")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="Checkpoint every N optimiser steps instead of once "
                         "per epoch. 0 keeps the per-epoch default. USE THIS "
                         "for any run over ~1h, and always for a single-epoch "
                         "run, where the per-epoch default writes nothing until "
                         "the very end and a crash costs the whole run.")
    ap.add_argument("--eval-steps", type=int, default=0,
                    help="Eval interval when --save-steps is set; defaults to "
                         "the same value.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03,
                    help="Converted to warmup_steps before it reaches "
                         "TrainingArguments; transformers 5.x removed the "
                         "ratio form.")
    ap.add_argument("--kbit-prep", choices=("auto", "peft", "norms-only"),
                    default="auto",
                    help="auto uses peft's blanket fp32 upcast when it fits in "
                         "60%% of free VRAM and falls back to upcasting only "
                         "1-D params. Recorded in the manifest because it is a "
                         "difference between runs.")
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto",
                    help="auto picks bf16 on compute capability >=8.0 and fp16 "
                         "below it. Turing (sm_75, the 2080 Ti) reports "
                         "is_bf16_supported()==True but has no bf16 tensor-core "
                         "path: measured 7.2 TFLOPS bf16 against 43.4 fp16, so "
                         "bf16 there is a 6x slowdown, not a correctness issue.")
    a = ap.parse_args()

    import torch
    from datasets import concatenate_datasets, load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    parts = []
    for p in a.train:
        d = load_from_disk(str(ROOT / p) if not Path(p).is_absolute() else p)
        if hasattr(d, "keys") and not hasattr(d, "column_names"):
            d = d[list(d.keys())[0]]
        parts.append(d.select_columns(["text"]))
        print(f"  {p}: {len(d):,} rows")
    ds = concatenate_datasets(parts).shuffle(seed=a.seed)
    print(f"pooled: {len(ds):,} rows")

    def tok_fn(b):
        return tok(b["text"], add_special_tokens=True)

    toked = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                   desc="tokenizing")

    # Pack into fixed blocks. Concatenating then slicing wastes nothing, which
    # matters when the whole corpus is under 2M tokens.
    # Gemma's tokenizer_config sets add_bos_token=False, so add_special_tokens
    # adds NOTHING and packed blocks start mid-stream with no BOS. Gemma is
    # extremely BOS-dependent: on Gemma-SEA-LION-v4.5-E2B, blocks of Cirebonese
    # score loss 11.22 (ppl 74,836 -- worse than uniform over its 262k vocab)
    # without one and 4.89 (ppl 133) with one at the head of every block. The
    # chat template hard-codes <bos>, which is why chat generation looked fine
    # while raw-text loss was garbage.
    # auto = on when the tokenizer HAS a bos but does not add it, so the
    # Llama/Qwen runs (add_bos_token=True) keep their original packing.
    _bos = tok.bos_token_id
    # PROBE the tokenizer, do not trust the attribute. Llama-SEA-LION exposes no
    # add_bos_token attribute at all on the fast tokenizer, yet DOES prepend
    # 128000 -- so reading the attribute with a False default concluded the
    # opposite of the truth and turned per-block BOS on for a tokenizer that
    # already had it. Gemma is the reverse: attribute present and False, and it
    # genuinely adds nothing. Only the behaviour distinguishes them.
    _adds_bos = False
    if _bos is not None:
        try:
            _adds_bos = tok("probe", add_special_tokens=True)["input_ids"][0] == _bos
        except Exception:
            _adds_bos = bool(getattr(tok, "add_bos_token", False))
    if a.bos_per_block == "auto":
        _bos_per_block = _bos is not None and not _adds_bos
    else:
        _bos_per_block = a.bos_per_block == "on"
    print(f"bos_token_id={_bos} tokenizer_prepends_bos={_adds_bos} "
          f"(attr={getattr(tok, 'add_bos_token', '<unset>')}) -> "
          f"bos_per_block={_bos_per_block}")

    def group(b):
        ids = []
        for x in b["input_ids"]:
            ids.extend(x)
        if _bos_per_block:
            body = a.seq_len - 1
            n = (len(ids) // body) * body
            blocks = [[_bos] + ids[i:i + body] for i in range(0, n, body)]
        else:
            n = (len(ids) // a.seq_len) * a.seq_len
            blocks = [ids[i:i + a.seq_len] for i in range(0, n, a.seq_len)]
        return {"input_ids": blocks, "labels": [x[:] for x in blocks]}

    packed = toked.map(group, batched=True, batch_size=1000,
                       remove_columns=toked.column_names, desc="packing")
    ntok = len(packed) * a.seq_len
    print(f"packed: {len(packed):,} blocks of {a.seq_len} = {ntok/1e6:.2f}M tokens")

    split = packed.train_test_split(test_size=a.val_frac, seed=a.seed)
    train_ds, val_ds = split["train"], split["test"]

    # transformers 4.55 takes torch_dtype; `dtype` is a newer alias and raises
    # a TypeError from inside the model constructor on this pinned version.
    # Gemma2 uses attention logit soft-capping that sdpa does not implement;
    # transformers warns to use eager for it. Not optional -- sdpa silently
    # computes different attention on this architecture.
    # AutoConfig, not a file read: --base may be a hub id, and reading
    # config.json off disk silently returns "" for one, which would pick sdpa
    # for a soft-capped architecture that needs eager.
    _arch = ""
    try:
        from transformers import AutoConfig
        _arch = " ".join(AutoConfig.from_pretrained(a.base).architectures or [])
    except Exception as _e:
        print(f"WARNING: could not read architectures ({type(_e).__name__}); "
              f"attention choice falls back to sdpa")
    # Gemma3/Gemma4 carry the same soft-capping in their text tower.
    _attn = "eager" if any(g in _arch for g in ("Gemma2", "Gemma3", "Gemma4")) \
        else "sdpa"

    # Turing has no bf16 tensor cores. torch.cuda.is_bf16_supported() still
    # returns True because bf16 *runs*, just on a slow emulated path -- so the
    # capability check is the real test, not the torch helper.
    _cap = torch.cuda.get_device_capability(0)
    if a.dtype == "auto":
        _dt = torch.bfloat16 if _cap >= (8, 0) else torch.float16
    else:
        _dt = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    _use_bf16 = _dt is torch.bfloat16
    print(f"sm_{_cap[0]}{_cap[1]} -> dtype={'bf16' if _use_bf16 else 'fp16'} "
          f"(--dtype {a.dtype}), attn={_attn}, arch={_arch or 'unknown'}")

    kw = dict(torch_dtype=_dt, device_map={"": 0}, attn_implementation=_attn)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dt,
            bnb_4bit_use_double_quant=True)
    # Multimodal checkpoints (Gemma4ForConditionalGeneration) are not in the
    # CausalLM auto-map; fall back and then take the language model, which is
    # what CPT on plain text should touch.
    try:
        model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM refused ({type(e).__name__}); "
              f"trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.base, **kw)
    _kbit_prep = None
    if a.load_4bit:
        # peft's prepare_model_for_kbit_training upcasts EVERY unquantised
        # parameter to fp32. bitsandbytes does not quantise embeddings, and
        # Gemma4 carries a per-layer input embedding table of 2.35B params
        # (259k vocab x 256 x 35 layers) on top of the usual 403M one -- so the
        # upcast asks for 11.07 GB on top of the quantised weights and dies on a
        # 16 GB card with "Failed to create GPU mapping".
        # The upcast exists for the numerical stability of norms and biases,
        # which are all 1-D. Doing just those costs a few MB and leaves the
        # embedding tables in bf16 where they were already fine.
        from peft import prepare_model_for_kbit_training
        _fp32_cost = sum(p.numel() for p in model.parameters()
                         if p.dtype in (torch.float16, torch.bfloat16)) * 4
        _free = torch.cuda.mem_get_info()[0]
        if a.kbit_prep == "peft" or (a.kbit_prep == "auto"
                                     and _fp32_cost < 0.6 * _free):
            _kbit_prep = "peft"
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True)
        else:
            _kbit_prep = "norms-only"
            print(f"kbit prep: blanket fp32 upcast would need "
                  f"{_fp32_cost/1e9:.2f} GB against {_free/1e9:.2f} GB free; "
                  f"upcasting 1-D params (norms/biases) only")
            for _n, _p in model.named_parameters():
                _p.requires_grad_(False)
                if _p.ndim == 1 and _p.dtype in (torch.float16, torch.bfloat16):
                    _p.data = _p.data.to(torch.float32)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"kbit prep = {_kbit_prep}")
    model.config.use_cache = False
    model.enable_input_require_grads()

    _PROJ = ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]
    # A multimodal checkpoint carries the same projection NAMES in its audio and
    # vision towers, and peft matches on the bare name -- so the plain list
    # adapts all three towers. On Gemma-SEA-LION-v4.5-E2B that is 148 adapted
    # modules (36 audio + 112 vision) that no text-only batch can ever put a
    # gradient through. Scope to the language model when there is one; keep the
    # bare list otherwise so the Llama/Qwen runs stay bit-for-bit reproducible.
    if hasattr(getattr(model, "model", None), "language_model"):
        _targets = (r"model\.language_model\..*\.(?:"
                    + "|".join(_PROJ) + r")$")
        print(f"multimodal base: scoping LoRA to model.language_model")
    else:
        _targets = _PROJ
    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=_targets)
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B "
          f"({100*trainable/total:.3f}%)")

    # transformers 5.x dropped warmup_ratio and kept only warmup_steps. Passing
    # the ratio there is a TypeError, and quietly dropping it would give the
    # 5.x runs no warmup at all -- a silent difference in the training recipe
    # between two runs meant to be matched. So convert the ratio to steps here,
    # reproducing HF's own ceil(total_steps * ratio), and pass steps on both
    # versions. Same schedule, no version dependence.
    _steps_per_epoch = max(len(train_ds) // a.batch // a.accum, 1)
    _total_steps = math.ceil(a.epochs * _steps_per_epoch)
    _warmup_steps = math.ceil(_total_steps * a.warmup_ratio)
    print(f"schedule: {_total_steps} optimiser steps "
          f"({_steps_per_epoch}/epoch x {a.epochs}), "
          f"warmup {_warmup_steps} (ratio {a.warmup_ratio})")

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        weight_decay=0.0, bf16=_use_bf16, fp16=not _use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=a.eval_batch,
        eval_accumulation_steps=1,
        logging_steps=10,
        # A SINGLE-EPOCH RUN SAVES ONLY AT THE END under save_strategy="epoch",
        # so a host that dies at 85% loses everything with nothing to --resume
        # from. That happened: taco went unreachable ~4h into a 1-epoch
        # Balinese CPT. Any run long enough to care about should checkpoint on
        # STEPS. save_total_limit=1 keeps the disk cost to one checkpoint.
        eval_strategy=("steps" if a.save_steps else "epoch"),
        save_strategy=("steps" if a.save_steps else "epoch"),
        **({"save_steps": a.save_steps, "eval_steps": a.eval_steps or a.save_steps}
           if a.save_steps else {}),
        save_total_limit=1, report_to=[], seed=a.seed,
        dataloader_num_workers=2, optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

    base_eval = trainer.evaluate()
    print(f"BASE eval loss {base_eval['eval_loss']:.4f} "
          f"ppl {math.exp(base_eval['eval_loss']):.1f}")

    trainer.train(resume_from_checkpoint=a.resume)
    final = trainer.evaluate()
    print(f"TUNED eval loss {final['eval_loss']:.4f} "
          f"ppl {math.exp(final['eval_loss']):.1f}")

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    summary = {
        "base": a.base, "train": a.train, "out": a.out,
        "blocks": len(packed), "seq_len": a.seq_len, "tokens": ntok,
        "lora": {"r": a.rank, "alpha": a.alpha, "dropout": a.dropout,
                 "trainable_M": round(trainable / 1e6, 2)},
        "load_4bit": a.load_4bit,
        "kbit_prep": _kbit_prep, "bos_per_block": _bos_per_block,
        "tokenizer_prepends_bos": _adds_bos,
        "save_steps": a.save_steps or None,
        "total_steps": _total_steps, "warmup_steps": _warmup_steps,
        "dtype": "bf16" if _use_bf16 else "fp16",
        "sm": f"{_cap[0]}{_cap[1]}",
        "lr": a.lr, "epochs": a.epochs,
        "eff_batch": a.batch * a.accum,
        "base_eval_loss": round(base_eval["eval_loss"], 4),
        "tuned_eval_loss": round(final["eval_loss"], 4),
        "base_ppl": round(math.exp(base_eval["eval_loss"]), 2),
        "tuned_ppl": round(math.exp(final["eval_loss"]), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(a.out, "cpt_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
