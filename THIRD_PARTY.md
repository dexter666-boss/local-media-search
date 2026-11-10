# Third-party notes

Dependencies and downloaded weights are not included in this archive. Their licenses, model cards, and any training-data conditions must be checked separately for the intended distribution/use; licensing our application does not relicense them.

- [OpenCLIP](https://github.com/mlfoundations/open_clip): multilingual image/text embeddings.
- [Multilingual CLIP model card](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k): default visual-search weights.
- [PyTorch / Torchvision](https://github.com/pytorch/vision): Faster R-CNN detectors and optional ConvNeXt Tiny dog verification.
- [Transformers](https://github.com/huggingface/transformers), [SentencePiece](https://github.com/google/sentencepiece), [OpenCV](https://github.com/opencv/opencv-python), [Pillow](https://github.com/python-pillow/Pillow): runtime libraries.
- [semantic-video-search](https://github.com/David-Ykz/semantic-video-search): inspiration. The locally downloaded reference archive has no LICENSE file. Do not copy its code or assets on an assumption of permission. An automated structural comparison is only a screening tool, not proof of provenance or legal clearance.

Application source is released under MIT; see LICENSE. Dependencies and model weights retain their own terms. No third-party authorization is implied by the application license.

- [RapidOCR 1.4.4](https://rapidai.github.io/RapidOCRDocs/v1.4.4/install_usage/api/RapidOCR/) and [ONNX Runtime](https://github.com/microsoft/onnxruntime): local OCR; default PP-OCRv4 recognition/detection and mobile text orientation weights are installed with the dependency package. Their files and notices are not bundled in this source archive.

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Whisper](https://github.com/openai/whisper), CTranslate2 and PyAV provide local speech transcription and audio decoding. Models/dependencies and their notices are downloaded separately and are not included in this archive.

- [opencc-python-reimplemented](https://github.com/yichen0831/opencc-python), Apache-2.0, supplies dictionary-based Chinese script conversion. Dependency files and upstream notices are installed separately.

- [cn2an](https://github.com/Ailln/cn2an), MIT, converts supported Chinese integer quantities; whole-sentence automatic conversion is not used.
