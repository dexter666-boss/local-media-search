# Release validation — 0.1.0

Date: 2026-11-08. Target: Windows, Python 3.12.14, CPU install.

## Passed

- Reproduced the historical 4.270-second speech end time on a 3.605-second input with a failing regression test; clipped bounds now pass and zero-length/out-of-range segments are discarded.
- Development regression: 94 tests passed.
- Installed the final runtime/test source into a newly created virtual environment using the README's two pip install commands. Package downloads could reuse pip caches. `pip check`: no broken requirements. All 94 tests passed in this environment.
- Actual OCR: two authored fictional notes, two queries, each returning only its expected file; original line highlight inspected in the UI. Second query reused both OCR caches.
- Actual faster-whisper small inference: a fictional system-synthesized video matched the expected phrase, a supported numeric variant, and rejected an absent phrase. Subsequent queries reused the transcript cache.
- Actual browser video playback and timestamp-button operation: media readyState 4, no media error. HTTP byte-range playback returned 206 and the requested 32 bytes.
- Actual multilingual CLIP inference in the new environment: encoded one image and one decoded video frame on CPU, produced two finite similarity values. This is a runtime check, not an accuracy benchmark.
- Release runtime/test Python files, requirements, launcher and UI matched the isolated environment; README local links and Python syntax checked. Package contents screened for private caches, environments, dataset media and common credential patterns.

The new environment installed sentencepiece 0.2.3 and timm 1.0.31, versus 0.2.2 and 1.0.30 in development. Both completed the above tests. Other tested core versions: torch 2.15.0+cpu, torchvision 0.29.2+cpu, open-clip-torch 3.3.0, transformers 4.57.6, rapidocr-onnxruntime 1.4.4, faster-whisper 1.2.1, ctranslate2 4.8.5, av 16.1.0, opencc-python-reimplemented 0.1.7 and cn2an 0.5.24.

## Not established

- Model weights were reused from local caches via validation-only directory junctions. No weights, junctions or environments are included in the source ZIP. First-ever model download is unverified.
- Another physical computer, other operating systems, CUDA in the new environment and hosted GitHub Actions are unverified.
- Functional fixtures are synthetic and do not measure natural-speech or general visual-search accuracy. Historical natural-recording results are described separately in EVALUATION.md.
