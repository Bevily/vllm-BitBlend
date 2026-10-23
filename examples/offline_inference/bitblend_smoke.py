# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a small deterministic generation check on local Qwen3 weights."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams, envs
from vllm.model_executor.kernels.linear.bitblend.sm120.database import (
    database_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tier", choices=("low", "mid", "high"))
    parser.add_argument("--backend", help="Registered BitBlend backend")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    hf_config = json.loads((model / "config.json").read_text())
    overrides = {}
    if args.backend and not args.tier:
        parser.error("--backend requires --tier")
    if args.tier:
        quant_config = dict(hf_config["quantization_config"])
        if quant_config["quant_method"] != "bitblend":
            raise ValueError("--tier requires a BitBlend checkpoint")
        quant_config["tier"] = args.tier
        if args.backend:
            quant_config["backend"] = args.backend
        overrides["quantization_config"] = quant_config
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    questions = [
        "What is 2 + 2? Answer with only the number.",
        "Translate into English: 你好。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in questions
    ]
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        hf_overrides=overrides,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=False,
        seed=42,
        attention_config={"backend": "TRITON_ATTN"},
    )
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))
    results = []
    for question, output in zip(questions, outputs):
        completion = output.outputs[0]
        if not completion.token_ids or not completion.text.strip():
            raise AssertionError("Empty generation")
        results.append(
            {
                "question": question,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
    report = {
        "model": str(model),
        "tier": args.tier or "bf16",
        "backend": quant_config["backend"] if args.tier else "unquantized",
        "tactic_database": database_provenance(
            envs.VLLM_BITBLEND_SM120_TACTICS,
            use_default=bool(args.tier and quant_config["backend"] != "reference"),
        ),
        "seed": 42,
        "max_tokens": 32,
        "results": results,
        "scope": "generation smoke check, not a quality benchmark",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
