#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:46:00 2026

@author: syed
"""

from typing import List, Dict
import json
import os
import textwrap
import requests

class LLMCandidateGenerator:
    def __init__(self, base_url: str, model: str, cache_dir: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _prompt_for_class(self, cls: str, seed_terms: List[str]) -> str:
        seed_sample = ", ".join(seed_terms[:20])  # ↓ 20 seeds max for prompt
        return textwrap.dedent(
            f"""You are an agricultural terminology expert. Generate 300+ candidate terms (synonyms, hyponyms, related phrases) for the {cls} class using these 20 seed examples: {seed_sample}

Rules:
- Output ONLY comma-separated terms, 1 per line, no explanations, no numbering
- Include multi-word phrases (2-5 words): "anaerobic digestion", "batch bioreactor"
- Vary word order/forms: "digestion anaerobic", "anaerobic digesters" 
- Add prefixes/suffixes: "enhanced biogas", "pilot-scale fermentation", "thermophilic composting"
- Generate 300-500 terms total

Examples: 
anaerobic bioreactor, biogas production, aerobic composting, batch fermentation, vermicomposting reactor

TERMS ONLY:"""
        ).strip()

    def _call_llm(self, seeds: List[str]) -> List[str]:  # Changed: seeds → List[str]
        print(f"Calling LLM for {len(seeds)} seeds... (target: 300+ terms)")
        prompt = self._prompt_for_class("TM", seeds)  # Generic TM prompt works for all
        
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,  # ↑ More tokens = more terms
            },
            timeout=600,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        
        # Parse comma-separated terms (robust)
        terms = []
        for line in content.splitlines():
            line = line.strip("•-• \t\n•-")
            if ',' in line:
                terms.extend([t.strip() for t in line.split(',') if len(t.strip()) > 2])
            elif line and len(line) > 2:
                terms.append(line)
        
        # Dedupe + filter
        uniq_terms = list(dict.fromkeys(terms))
        filtered = [t for t in uniq_terms if 2 <= len(t) <= 80]
        
        print(f"Generated {len(filtered)} terms")
        return filtered[:500]  # Cap at 500

    def _parse_terms(self, raw: List[str]) -> List[str]:
        # Simplified - raw is already list
        return raw

    def generate_for_class(self, cls: str, seed_terms: List[str]) -> List[str]:
        cache_path = os.path.join(self.cache_dir, f"{cls}_llm_candidates.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Generate
        candidates = self._call_llm(seed_terms)  # Pass seeds directly
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        return candidates

