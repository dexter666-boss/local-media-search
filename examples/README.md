# Small reproducible example / 小样例

These are fictional functional fixtures, not an accuracy benchmark. No personal chat or natural-recording corpus is included.

## Screenshot text / 图片文字

1. Start the app and select “文件夹找素材” → “找图片文字”.
2. Browse to this repository's `examples/screenshots` folder and scan.
3. Search `带上电脑`: only `meeting-note.png` should match.
4. Search `周六上午`: only `weekend-note.png` should match. Preview to inspect the highlighted original line.

## Spoken line / 视频台词

1. Select “找讲话” and scan `examples/media`.
2. Search `带上电脑`. First use downloads the speech model (~0.5 GB) and transcribes the video.
3. Expand the candidate and click its timestamp to play. Approximate timings vary with transcription; they are not exact cut boundaries.

The video contains system-synthesized fictional Chinese speech, with initial silence. The recording includes: “明天下午三点开会，请记得带上电脑。” Listen to confirm; exact transcription is not guaranteed.

The UI captures in the READMEs come from actual runs of these fixtures. No model outputs are hard-coded into the application.
