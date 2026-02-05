#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:22:01 2026

@author: syed
"""

from typing import Dict, List, Tuple
import pandas as pd

Category = str  # "TM", "OWT", "AV"

def build_classification_table(
    tm_variants: Dict[str, List[str]],
    owt_variants: Dict[str, List[str]],
    av_variants: Dict[str, List[str]],
) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []

    # Seed terms
    for t in tm_variants.keys():
        rows.append((t, "TM"))
    for t in owt_variants.keys():
        rows.append((t, "OWT"))
    for t in av_variants.keys():
        rows.append((t, "AV"))

    # Variants
    for t, vars_ in tm_variants.items():
        for v in vars_:
            if v != t:
                rows.append((v, "TM+"))
    for t, vars_ in owt_variants.items():
        for v in vars_:
            if v != t:
                rows.append((v, "OWT+"))
    for t, vars_ in av_variants.items():
        for v in vars_:
            if v != t:
                rows.append((v, "AV+"))

    df = pd.DataFrame(rows, columns=["term", "label"])
    df = df.drop_duplicates()

    # Optional: if a term falls into several classes, you may:
    # - keep all rows, or
    # - aggregate labels into a list per term.
    # Example for aggregation:
    # df = df.groupby("term")["label"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()

    return df
