#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:24:02 2026

@author: syed
"""

import argparse
import os
from agro_terminology.config import Paths, LLMConfig
from agro_terminology.pipeline import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract agroecological terminologies and classify into TM/TM+/OWT/OWT+/AV/AV, optionally using LLM candidates."
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--tm-seed", required=True)
    p.add_argument("--owt-seed", required=True)
    p.add_argument("--av-seed", required=True)
    p.add_argument("--outdir", required=True)

    # LLM options
    p.add_argument("--use-llm", action="store_true", help="Use LLM to expand seed terms.")
    p.add_argument("--llm-base-url", default="http://localhost:1234")
    p.add_argument("--llm-model", default="mistral-7b-instruct")

    return p.parse_args()

def main():
    args = parse_args()
    paths = Paths(
        corpus=args.corpus,
        tm_seed=args.tm_seed,
        owt_seed=args.owt_seed,
        av_seed=args.av_seed,
        output_dir=args.outdir,
    )

    llm_cfg = None
    if args.use_llm:
        llm_cfg = LLMConfig(
            base_url=args.llm_base_url,
            model=args.llm_model,
            use_llm=True,
        )

    out_path = run_pipeline(paths, llm_cfg=llm_cfg)
    print(f"Saved labeled terminology file to: {out_path}")

if __name__ == "__main__":
    main()
