# hover_data.py
from __future__ import annotations

import random
from datasets import load_dataset
import dspy


def _extract_gold_titles(x) -> list[str]:
    """
    Robustly extract gold wiki titles from HoVer-style examples.

    Handles:
      - supporting_facts: List[[title, sent_id], ...]
      - supporting_facts: List[{"key": title, ...}, ...]  (some parquet conversions)
    """
    sf = x.get("supporting_facts", []) or []
    titles: list[str] = []

    for item in sf:
        t = None

        if isinstance(item, dict):
            # parquet conversions sometimes use {"key": "...", "value": ...}
            t = item.get("key") or item.get("title") or item.get("wiki_title")

        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            # canonical Hotpot/HoVer style: [title, sent_id]
            t = item[0]

        if isinstance(t, str) and t.strip():
            titles.append(t.strip())

    # De-dupe, keep order
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_hover_splits(
    seed: int = 0,
    n_train: int = 150,
    n_dev: int = 300,
    n_test: int = 300,
    *,
    require_num_hops: int | None = None,
    max_num_hops: int | None = 3,
):
    """
    Loads HoVer examples and creates deterministic 150/300/300 splits.

    - require_num_hops: only keep examples with exactly this num_hops (optional)
    - max_num_hops: only keep examples with num_hops <= this (default: 3, matching "up to 3-hop")
    """
    ds = load_dataset("vincentkoc/hover-parquet", split="train")

    pool = []
    seen_hpqa = set()

    for x in ds:
        hops = x.get("num_hops", None)

        if require_num_hops is not None and hops != require_num_hops:
            continue
        if max_num_hops is not None and isinstance(hops, int) and hops > max_num_hops:
            continue

        hid = x.get("hpqa_id")
        if hid and hid in seen_hpqa:
            continue
        if hid:
            seen_hpqa.add(hid)

        titles = _extract_gold_titles(x)
        if not titles:
            continue

        pool.append(
            dspy.Example(
                claim=x["claim"],
                titles=titles,
                hpqa_id=hid,
                num_hops=hops,
            ).with_inputs("claim")
        )

    rng = random.Random(seed)
    rng.shuffle(pool)

    need = n_train + n_dev + n_test
    if len(pool) < need:
        raise ValueError(f"Not enough usable examples: have {len(pool)}, need {need}")

    train = pool[:n_train]
    dev   = pool[n_train:n_train + n_dev]
    test  = pool[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test
