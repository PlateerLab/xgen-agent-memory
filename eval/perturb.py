"""Korean robustness perturbations — 6 axes, deterministic and seeded.

Each generator takes a clean query and returns a perturbed variant (or None
when the query has no applicable site). Variants share the parent query's
qrels, so every axis is scored as a DELTA against the same clean queries —
isolating robustness from base difficulty.

Axes (KOMBO-style typo protocol + Korean search-engineering failure modes):
    josa      trailing 조사 swap (은↔는, 이↔가, 을↔를, …)
    spacing   remove every space (붙여쓰기)
    jamo      1–2 jamo-level edits (substitute/delete tail/transpose)
    compound  keep only the longest noun-ish token (복합명사 부분 질의)
    koen      swap common loanwords to English (도커→docker)
    homonym   strip disambiguating context to a bare ambiguous head
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgen_agent_memory.hangul import (  # noqa: E402
    _BASE, is_hangul_syllable,
)

SEED = 41

_JOSA_SWAP = {
    "은": "는", "는": "은", "이": "가", "가": "이", "을": "를", "를": "을",
    "에서": "에", "으로": "로", "와": "과", "과": "와",
}

_KOEN = {
    "도커": "docker", "쿠버네티스": "kubernetes", "파이썬": "python",
    "리눅스": "linux", "데이터베이스": "database", "서버": "server",
    "컴퓨터": "computer", "인터넷": "internet", "프로그램": "program",
    "시스템": "system", "네트워크": "network", "메모리": "memory",
    "게임": "game", "테스트": "test", "코드": "code", "웹": "web",
    "미국": "USA", "영국": "UK",
}


def _decompose(ch: str):
    idx = ord(ch) - _BASE
    return idx // 588, (idx % 588) // 28, idx % 28


def _compose(lead: int, vowel: int, tail: int) -> str:
    return chr(_BASE + (lead * 21 + vowel) * 28 + tail)


def perturb_josa(q: str, rng: random.Random) -> Optional[str]:
    words = q.split()
    sites = []
    for i, w in enumerate(words):
        for josa, repl in _JOSA_SWAP.items():
            if len(w) > len(josa) + 1 and w.endswith(josa):
                sites.append((i, josa, repl))
                break
    if not sites:
        return None
    i, josa, repl = rng.choice(sites)
    words[i] = words[i][: -len(josa)] + repl
    return " ".join(words)


def perturb_spacing(q: str, rng: random.Random) -> Optional[str]:
    if " " not in q:
        return None
    return q.replace(" ", "")


def perturb_jamo(q: str, rng: random.Random) -> Optional[str]:
    sites = [i for i, ch in enumerate(q) if is_hangul_syllable(ch)]
    if not sites:
        return None
    out = list(q)
    for i in rng.sample(sites, k=min(2, len(sites)))[:rng.randint(1, 2)]:
        lead, vowel, tail = _decompose(out[i])
        op = rng.choice(["sub_vowel", "del_tail", "sub_lead"])
        if op == "sub_vowel":
            vowel = (vowel + rng.randint(1, 20)) % 21
        elif op == "del_tail" and tail:
            tail = 0
        else:
            lead = (lead + rng.randint(1, 18)) % 19
        out[i] = _compose(lead, vowel, tail)
    res = "".join(out)
    return res if res != q else None


def perturb_compound(q: str, rng: random.Random) -> Optional[str]:
    words = [w for w in q.split() if len(w) >= 2 and is_hangul_syllable(w[0])]
    if len(words) < 2:
        return None
    longest = max(words, key=len)
    return longest if longest != q.strip() else None


def perturb_koen(q: str, rng: random.Random) -> Optional[str]:
    out = q
    hit = False
    for ko, en in _KOEN.items():
        if ko in out:
            out = out.replace(ko, en)
            hit = True
    return out if hit else None


def perturb_homonym(q: str, rng: random.Random) -> Optional[str]:
    # Bare-head query: keep the first two eojeol only (strips the
    # disambiguating tail context).
    words = q.split()
    if len(words) < 4:
        return None
    return " ".join(words[:2])


AXES: Dict[str, Callable[[str, random.Random], Optional[str]]] = {
    "josa": perturb_josa,
    "spacing": perturb_spacing,
    "jamo": perturb_jamo,
    "compound": perturb_compound,
    "koen": perturb_koen,
    "homonym": perturb_homonym,
}


def main() -> None:
    data = Path(__file__).parent / "data"
    out_dir = data / "perturbations"
    out_dir.mkdir(exist_ok=True)
    queries: List[dict] = [json.loads(line) for line in
                           (data / "queries_clean.jsonl").open(encoding="utf-8")]
    for axis, fn in AXES.items():
        rng = random.Random(SEED)
        rows = []
        for q in queries:
            variant = fn(q["text"], rng)
            if variant and variant != q["text"]:
                rows.append({"qid": q["qid"], "axis": axis, "text": variant})
        with (out_dir / f"{axis}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{axis}: {len(rows)} variants")


if __name__ == "__main__":
    main()
