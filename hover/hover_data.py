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
    require_unique_docs: int | None = 3,  # Filter by unique gold doc count (LangProbe style)
):
    """
    Loads HoVer examples and creates deterministic 150/300/300 splits.

    Following LangProbe implementation:
    - Filters by count of UNIQUE gold documents (not num_hops metadata field)
    - require_unique_docs=3 means exactly 3 unique supporting documents
    - This matches the GEPA paper benchmark setup

    - require_unique_docs: only keep examples with exactly this many unique gold docs (default: 3)
    """
    # Use HoVer dataset from HuggingFace
    # Note: Official hover-nlp/hover uses deprecated loading scripts
    # Using parquet version which has the same data in standard format
    try:
        # Try parquet version first (standard format, no scripts)
        ds = load_dataset("vincentkoc/hover-parquet", split="train")
    except Exception as e:
        print(f"[HoVer] Warning: Could not load parquet version: {e}")
        # Fallback: try to download and use local data
        raise RuntimeError(
            "HoVer dataset loading failed. The official hover-nlp/hover dataset uses "
            "deprecated loading scripts. Please use vincentkoc/hover-parquet or "
            "download the data manually from https://hover-nlp.github.io/"
        )

    pool = []
    seen_hpqa = set()

    for x in ds:
        hid = x.get("hpqa_id")
        if hid and hid in seen_hpqa:
            continue
        if hid:
            seen_hpqa.add(hid)

        titles = _extract_gold_titles(x)
        if not titles:
            continue

        # Filter by unique gold document count (matching LangProbe's count_unique_docs)
        if require_unique_docs is not None and len(titles) != require_unique_docs:
            continue

        pool.append(
            dspy.Example(
                claim=x["claim"],
                titles=titles,
                hpqa_id=hid,
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
