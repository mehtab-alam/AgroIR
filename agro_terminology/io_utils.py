#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:22:30 2026

@author: syed
"""

from typing import List
import pandas as pd

def load_corpus(path: str) -> List[str]:
    df = pd.read_csv(path, encoding="latin1", sep=";")
    # Notebook: text := Title + " " + Abstract [file:1]
    titles = df["Title"].fillna("").astype(str).tolist()
    abstracts = df["Abstract"].fillna("").astype(str).tolist()
    return [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]

def load_seed_terms(path: str) -> List[str]:
    df = pd.read_csv(path, header=None, encoding="utf-8-sig")
    return [str(x).strip() for x in df[0].tolist() if isinstance(x, str) and x.strip()]
