"""Tokenizer: IndicF5's char tokenizer without jieba/pypinyin.

IndicF5 runs every prompt through f5_tts.model.utils.convert_char_to_pinyin (jieba word cut +
pypinyin) before the char->id lookup. For text with no CJK characters that pipeline reduces to:
  * translate ``; “ ” ‘ ’`` -> ``, " " ' '``,
  * one token per character,
  * plus a space inserted before every multi-character ASCII segment (jieba yields runs of
    [A-Za-z0-9] as one segment) unless the previous token is one of `` :'"``.
That is what ``tokenize`` does. ``tests/test_text.py`` checks it against the real function.
CJK input is rejected rather than silently mis-tokenized.
"""
from __future__ import annotations

import re

_TRANS = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})
# jieba: ASCII alnum (+ `+#&._%-`) blocks go through the DAG/HMM cut, whose finalseg yields
# [a-zA-Z0-9]+(\.\d+)?%? runs whole and everything else one char at a time.
_HAN_BLOCK = re.compile(r"([一-鿕a-zA-Z0-9+#&\._%\-]+)")
_ENG_RUN = re.compile(r"[a-zA-Z0-9]+(?:\.\d+)?%?|.", re.S)
_CJK = re.compile(r"[㄀-鿿]")


def load_vocab(path: str) -> dict[str, int]:
    """Exactly f5_tts get_tokenizer(..., 'custom'): token = line[:-1], id = line number."""
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line[:-1]] = i
    return vocab


def _segments(text: str):
    for blk in _HAN_BLOCK.split(text):
        if not blk:
            continue
        if _HAN_BLOCK.fullmatch(blk):
            yield from _ENG_RUN.findall(blk)
        else:
            yield from blk  # jieba yields non-han, non-ascii-word chars one by one


def to_char_list(text: str) -> list[str]:
    if _CJK.search(text):
        raise ValueError("CJK text is not supported by the MLX tokenizer")
    out: list[str] = []
    for seg in _segments(text.translate(_TRANS)):
        nbytes = len(seg.encode("utf-8"))
        if nbytes == len(seg):  # pure ASCII segment
            if out and nbytes > 1 and out[-1] not in " :'\"":
                out.append(" ")
            out.extend(seg)
        else:
            out.extend(seg)
    return out


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(c, 0) for c in to_char_list(text)]
