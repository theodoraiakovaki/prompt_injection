# Data cleaning

Run [promptshield_cleaning.ipynb](promptshield_cleaning.ipynb) from the repository root or this folder. It reads the published source splits in `../data/` and writes only the six JSON files below.

| Variant | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `data/exact/` | 18,909 | 1,000 | 23,359 |
| `data/normalized/` | 18,830 | 1,000 | 23,359 |

Both remove invalid rows, within-split exact label conflicts, exact duplicates, and exact overlap (test > validation > train). Normalized additionally removes normalized train conflicts/duplicates and train overlap with evaluation, then lowercases all splits. Its comparison key ignores case and repeated whitespace, preserving punctuation.

Evaluation rows and labels match across variants; normalized uses lowercase text. In the root analysis notebook, select `Path("cleaning/data/exact")` or `Path("cleaning/data/normalized")` and retrain from scratch with the same settings. These are derived splits, separate from official-split results.

Data provenance and license: [source README](../data/README.md), [Apache 2.0](../data/LICENSE). The unredacted backup in `../data/originals/` is local only and excluded from Git.
