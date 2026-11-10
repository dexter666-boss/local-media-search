# Evaluation scope / 评测说明

## Application behavior

The v0.1.0 source passes 94 unit/integration tests in the development environment. Tests cover routing, OCR matching, literal speech retrieval, cache behavior, HTTP handling, playback endpoints and timestamp-boundary regression. Model inference is replaced in several tests, so this count is not an accuracy claim.

## Historical natural-recording test: third batch

Before inference, references and queries were frozen for 100 Chinese AISHELL clips and 100 English RAVDESS videos, with no file-hash overlap with the two earlier batches. Chinese samples came from the same corpus; Chinese speaker disjointness was not established. English videos used four new actors but only two repeated sentences. Public training-data overlap is unknown.

| Measure | Chinese audio, 100 clips | English video, 100 clips |
|---|---:|---:|
| Raw normalized transcription error | CER 20.17% | WER 0.50% |
| Exact normalized transcript | 23/100 | 98/100 |
| Full-sentence retrieval precision / recall | 100% / 35% | 100% / 99% |
| Fixed keyword retrieval precision / recall | 100% / 100% (10 queries) | 100% / 99.75% (7 queries) |

Chinese fixed substrings of up to six characters: precision 100%, recall 67% over 100 queries. Keyword results do not establish full-sentence accuracy, cross-corpus generalization or performance on noisy recordings. Precision/recall aggregate query-file pairs against frozen transcript references; no combined Chinese/English headline score is reported.

Sources: [AISHELL-1](https://www.openslr.org/33/), [processed AISHELL](https://huggingface.co/datasets/urarik/AISHELL-1-processed), [RAVDESS](https://zenodo.org/records/1188976). Original dataset media is not distributed here.

One estimated English timestamp exceeded its audio duration by approximately 0.665 seconds. v0.1.0 fixes this with clipping plus a regression test. The historical corpus was subsequently deleted at the owner's request to free disk space, so these accuracy figures are historical and were not rerun after the boundary-only fix. No word-level timing truth was available: this is not a precise-localization benchmark.

Current repository examples are synthetic functional checks. They must not be used to claim natural-speech, general visual-search or OCR accuracy. Broader, reproducible cross-corpus evaluation remains future work.
