#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for NER
"""

from util import one_hot

LBLS = ["ANS","NOT"]
NONE = "NOT"
LMAP = {k: one_hot(2,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 50
