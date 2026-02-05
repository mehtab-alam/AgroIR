#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:23:37 2026

@author: syed
"""

import os
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm
import nltk
from typing import Optional
from .config import Paths, LLMConfig
from .llm_candidates import LLMCandidateGenerator


from .config import Paths
from .io_utils import load_corpus, load_seed_terms
from .indexing import build_corpus_tokens, build_normalised_index
from .variation_extraction import extract_variants_for_seed_list
from .classification import build_classification_table



def ensure_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("wordnet", quiet=True)

def _extract_for_category(
    category: str,
    seeds: List[str],
    norm_index: Dict[str, List[int]],
    corpus_words: List[str],
    cache_dir: str,
) -> Dict[str, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    json_path = os.path.join(cache_dir, f"{category}_variants.json")
    csv_path = os.path.join(cache_dir, f"{category}_variants_pairs.csv")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            variants = json.load(f)
        return variants

    variants: Dict[str, List[str]] = {}
    for term in tqdm(seeds, desc=f"Extracting variants for {category}"):
        if term in variants:
            continue
        # variation_extraction already re-cleans terms; keep same behaviour as notebook
        variants_for_seed = extract_variants_for_seed_list([term], norm_index, corpus_words)
        variants.update(variants_for_seed)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(variants, f, ensure_ascii=False, indent=2)

    # also write (seed, variant) pairs like in notebook
    rows = []
    for seed, vals in variants.items():
        for v in vals:
            if v != seed:
                rows.append((seed, v))
    df_pairs = pd.DataFrame(rows, columns=["seed", "variant"])
    df_pairs.to_csv(csv_path, index=False, encoding="utf-8")

    return variants

def run_pipeline(paths: Paths, k: int = 2, llm_cfg: Optional[LLMConfig] = None) -> str:
    ensure_nltk()

    corpus_texts = load_corpus(paths.corpus)
    words, tags = build_corpus_tokens(corpus_texts)
    norm_index = build_normalised_index(words, tags)

    tm_seeds = load_seed_terms(paths.tm_seed)
    owt_seeds = load_seed_terms(paths.owt_seed)
    av_seeds = load_seed_terms(paths.av_seed)

    # --- NEW: LLM expansion of seed sets ---
    if llm_cfg is not None and llm_cfg.use_llm:
        cache_dir = os.path.join(paths.output_dir, "llm_cache")
        gen = LLMCandidateGenerator(llm_cfg.base_url, llm_cfg.model, cache_dir)

        tm_llm = gen.generate_for_class("TM", tm_seeds)
        owt_llm = gen.generate_for_class("OWT", owt_seeds)
        av_llm = gen.generate_for_class("AV", av_seeds)

        # Merge and deduplicate, keep cap to avoid explosion
        def merge(base, extra, max_add=300):
            base_set = set(base)
            add = [t for t in extra if t not in base_set]
            return base + add[:max_add]

        tm_seeds = merge(tm_seeds, tm_llm)
        owt_seeds = merge(owt_seeds, owt_llm)
        av_seeds = merge(av_seeds, av_llm)

    cache_dir = os.path.join(paths.output_dir, "intermediate")
    tm_variants = _extract_for_category("TM", tm_seeds, norm_index, words, cache_dir)
    owt_variants = _extract_for_category("OWT", owt_seeds, norm_index, words, cache_dir)
    av_variants = _extract_for_category("AV", av_seeds, norm_index, words, cache_dir)

    df_terms = build_classification_table(tm_variants, owt_variants, av_variants)

    os.makedirs(paths.output_dir, exist_ok=True)
    out_path = os.path.join(paths.output_dir, "agro_terminologies_labeled_llm.csv" if llm_cfg and llm_cfg.use_llm else "agro_terminologies_labeled.csv")
    df_terms.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
