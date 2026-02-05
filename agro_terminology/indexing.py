#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:20:41 2026

@author: syed
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from .text_normalization import pos_tag_document, normalize_word

def build_corpus_tokens(corpus_texts: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Flatten all documents into a single words/tags list in notebook style."""
    tagged_sentences_all = []
    for doc in corpus_texts:
        tagged_sentences_all.extend(pos_tag_document(doc))
    tagged_tokens = [(token, tag) for sent in tagged_sentences_all for (token, tag) in sent]
    words = [w for (w, _) in tagged_tokens]
    tags = tagged_tokens
    return words, tags

def build_normalised_index(
    words: List[str], tags: List[Tuple[str, str]]
) -> Dict[str, List[int]]:
    """Inverse index: normalized root -> positions in flattened `words` list."""
    normalised = defaultdict(list)
    for idx, (word, tag) in enumerate(tags):
        norm = normalize_word(word, tag)
        normalised[norm].append(idx)
    return dict(normalised)
