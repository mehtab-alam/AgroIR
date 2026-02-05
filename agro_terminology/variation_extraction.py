#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:21:23 2026

@author: syed
"""

from typing import Dict, List, Tuple
from .text_normalization import pos_tag_text, normalize_word, clean_text

def _matching(pos1: List[int], pos2: List[int], k: int) -> List[Tuple[int, int]]:
    pos2_set = set(pos2)
    matches: List[Tuple[int, int]] = []
    for p in pos1:
        for i in range(p - k, p + k + 1):
            if i in pos2_set:
                matches.append((min(p, i), max(p, i)))
                break
    return matches

def extract_single_variants(
    terms: List[str],
    norm_index: Dict[str, List[int]],
    corpus_words: List[str],
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for term in set(terms):
        tagged = pos_tag_text(term)
        if not tagged:
            results[term] = []
            continue
        word, tag = tagged[0]
        root = normalize_word(word, tag)
        if root in norm_index:
            variants = []
            for i in norm_index[root]:
                v = corpus_words[i]
                if v != term:
                    variants.append(v)
            results[term] = list(set(variants))
        else:
            results[term] = []
    return results

def extract_couple_variants(
    terms: List[str],
    norm_index: Dict[str, List[int]],
    corpus_words: List[str],
    k: int = 2,
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for term in set(terms):
        tokens = term.split()
        if len(tokens) != 2:
            continue
        tagged = pos_tag_text(term)
        if len(tagged) != 2:
            results[term] = []
            continue
        (w1, t1), (w2, t2) = tagged
        r1 = normalize_word(w1, t1)
        r2 = normalize_word(w2, t2)
        if r1 in norm_index and r2 in norm_index:
            matches = _matching(norm_index[r1], norm_index[r2], k)
            variants = [" ".join(corpus_words[a:b + 1]) for a, b in matches]
            results[term] = list(set(variants))
        else:
            results[term] = []
    return results

def extract_variants_for_seed_list(
    seed_terms: List[str],
    norm_index: Dict[str, List[int]],
    corpus_words: List[str],
    k: int = 2,
) -> Dict[str, List[str]]:
    """Notebook behaviour: if 2‑word → couple variants, else single variants."""
    seed_clean = [clean_text(t) for t in seed_terms if t and isinstance(t, str)]
    singles = [t for t in seed_clean if len(t.split()) == 1]
    couples = [t for t in seed_clean if len(t.split()) == 2]
    res: Dict[str, List[str]] = {}
    res.update(extract_single_variants(singles, norm_index, corpus_words))
    res.update(extract_couple_variants(couples, norm_index, corpus_words, k=k))
    return res
