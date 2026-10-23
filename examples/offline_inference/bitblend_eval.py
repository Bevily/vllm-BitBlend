# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate one fixed BitBlend tier or its BF16 baseline on WikiText."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tier", choices=("low", "mid", "high"))
    parser.add_argument("--backend", default="auto", help="Registered BitBlend backend")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--limit", type=int, help="Document count; omit for full test set"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.max_model_len < 2 or (args.limit is not None and args.limit < 1):
        parser.error("Context length must be >= 2 and document limit must be positive")
    if args.report.exists():
        parser.error("Report already exists; select a new output path")

    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable

    from vllm import envs
    from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
        database_provenance,
    )

    model = args.model.resolve()
    config = json.loads((model / "config.json").read_text())
    quant_config = config.get("quantization_config")
    overrides = {}
    if args.tier:
        if not quant_config or quant_config.get("quant_method") != "bitblend":
            parser.error("--tier requires a BitBlend checkpoint")
        overrides["quantization_config"] = {
            **quant_config,
            "tier": args.tier,
            "backend": args.backend,
        }
    elif quant_config:
        parser.error("BF16 baseline requires an unquantized checkpoint")

    model_args = {
        "pretrained": str(model),
        "tokenizer": str(model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.5,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "seed": 42,
        "add_bos_token": True,
        "hf_overrides": overrides,
        "attention_config": {"backend": "TRITON_ATTN"},
    }
    task = TaskManager().load("wikitext")["tasks"]["wikitext"]
    dataset = task.test_docs()
    dataset_metadata = {
        "path": task.config.dataset_path,
        "name": task.config.dataset_name,
        "split": task.config.test_split,
        "fingerprint": dataset._fingerprint,
        "documents": len(dataset),
    }
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task],
        num_fewshot=0,
        batch_size=1,
        limit=args.limit,
        apply_chat_template=False,
        log_samples=False,
        bootstrap_iters=0,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )
    if results is None:
        raise RuntimeError("Evaluation returned no results")
    metrics = results["results"]["wikitext"]
    if not math.isfinite(metrics["word_perplexity,none"]):
        raise AssertionError("Non-finite WikiText perplexity")
    report = {
        "model": str(model),
        "tier": args.tier or "bf16",
        "backend": args.backend if args.tier else "unquantized",
        "tactic_database": database_provenance(
            envs.VLLM_BITBLEND_SM120_TACTICS,
            use_default=bool(args.tier and args.backend != "reference"),
        ),
        "scope": "full test set" if args.limit is None else "limited document check",
        "metric": "lm-eval WikiText word perplexity; not token perplexity",
        "model_args": model_args,
        "dataset": dataset_metadata,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lm_eval", "datasets", "vllm", "torch", "transformers")
        },
        "evaluation": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False, default=handle_non_serializable
        )
        + "\n"
    )
    print(json.dumps({"tier": report["tier"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
