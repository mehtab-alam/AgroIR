#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 00:20:13 2026

@author: syed
"""

# agro_terminology/text_normalization.py
import re
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import Iterable



stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^0-9A-Za-zÀ-ÿ\- ]", " ", text)
    text = re.sub(r"-+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def penn_to_wn_tag(tag: str) -> Optional[str]:
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    if tag.startswith("V"):
        return wn.VERB
    return None

def pos_tag_text(text: str) -> List[Tuple[str, str]]:
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)

def normalize_word(word: str, tag: str) -> str:
    word = word.lower()
    wntag = penn_to_wn_tag(tag)
    if wntag is not None:
        lemma = lemmatizer.lemmatize(word, pos=wntag)
        if lemma:
            return stemmer.stem(lemma)
    return stemmer.stem(word)


def pos_tag_document(document: str) -> List[List[Tuple[str, str]]]:
    """Sentence‑wise tokenization, global POS tagging, then reconstructed per sentence."""
    sentences = nltk.sent_tokenize(document)
    tokenized = [nltk.word_tokenize(sent) for sent in sentences]
    flat_tokens = [tok for sent in tokenized for tok in sent]
    flat_tagged = nltk.pos_tag(flat_tokens)
    tagged_sentences = []
    idx = 0
    for sent in tokenized:
        n = len(sent)
        tagged_sentences.append(flat_tagged[idx: idx + n])
        idx += n
    return tagged_sentences
