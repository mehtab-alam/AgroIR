#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:13:24 2026

@author: syed
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Paths:
    corpus: str
    tm_seed: str
    owt_seed: str
    av_seed: str
    output_dir: str

@dataclass
class LLMConfig:
    base_url: str          # e.g. "http://localhost:11434" or "http://localhost:1234"
    model: str             # e.g. "mistral-7b-instruct"
    use_llm: bool = False  # flag to enable / disable LLM expansion
