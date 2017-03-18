#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for NER
"""

from util import one_hot

LBLS = ["ANS_S", "ANS_E", "NOT"]
NONE = "NOT"
LMAP = {k: one_hot(3,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 50
