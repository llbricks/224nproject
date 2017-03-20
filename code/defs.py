#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for NER
"""

from util import one_hot

LBLS = [1, 2, 0]
NONE = 0
LMAP = {k: one_hot(3,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 50
