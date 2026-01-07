#!/usr/bin/env python3
"""
Rebuild the Hover BM25 index from scratch.
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, '/work1/krishnamurthy/arvind/adBO/hover')

from wiki_retriever import build_or_load_bm25, make_search_fn

work_dir = Path("/work1/krishnamurthy/arvind")
wiki_dir = work_dir / "wiki17"
index_dir = work_dir / "wiki17_bm25"

print("=" * 80)
print("Rebuilding Hover BM25 Index")
print("=" * 80)
print()

# Backup old index
old_index_dir = work_dir / "wiki17_bm25_OLD_BACKUP"
if index_dir.exists():
    print(f"Backing up old index to: {old_index_dir}")
    if old_index_dir.exists():
        shutil.rmtree(old_index_dir)
    shutil.move(str(index_dir), str(old_index_dir))
    print("✓ Backup complete")
    print()

# Build new index
print(f"Building new BM25 index at: {index_dir}")
print("This may take 5-10 minutes...")
print()

try:
    corpus, retriever, stemmer = build_or_load_bm25(wiki_dir=wiki_dir, index_dir=index_dir)
    print(f"✓ Successfully built index with {len(corpus):,} documents")
    print()
except Exception as e:
    print(f"✗ Failed to build index: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the new index
print("Testing new index...")
print("-" * 80)

search_fn = make_search_fn(corpus, retriever, stemmer, n_threads=4)

test_queries = [
    ("Florence Cathedral", ["Florence Cathedral"]),
    ("Gaddo Gaddi", ["Gaddo Gaddi"]),
    ("Barack Obama", ["Barack Obama", "Sarah Onyango Obama"]),
]

all_good = True
for query, expected_titles in test_queries:
    print(f"\nQuery: '{query}'")
    docs, scores = search_fn(query, k=10)
    titles = [doc.split(" | ", 1)[0].strip() for doc in docs]

    # Check if any expected title is in top 10
    found = any(exp in titles for exp in expected_titles)

    if found:
        matching = [t for t in titles if t in expected_titles]
        print(f"  ✓ PASS - Found: {matching}")
        print(f"    Top 3: {titles[:3]}")
    else:
        print(f"  ✗ FAIL - Expected one of {expected_titles}")
        print(f"    Got: {titles[:5]}")
        all_good = False

print()
print("=" * 80)

if all_good:
    print("✓ All tests passed! Index rebuilt successfully.")
    print()
    print("You can now re-run the Hover benchmark.")
else:
    print("✗ Some tests failed. There may still be an issue with the retriever.")

sys.exit(0 if all_good else 1)
