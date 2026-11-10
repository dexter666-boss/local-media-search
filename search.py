"""Learning implementation of text-to-video search. See README.md."""
import argparse
import json
import math
import statistics
import sys


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('expected a positive finite number')
    return result


def select_scenes(times, scores, duration, interval, minimum=.22, sigma=2):
    if not times or len(times) != len(scores):
        raise ValueError('timestamps and scores must be nonempty and equally sized')
    threshold = max(minimum, statistics.fmean(scores) + sigma * statistics.pstdev(scores))
    scenes = []
    for timestamp, score in zip(times, scores):
        if score < threshold:
            continue
        end = min(timestamp + interval, duration)
        if end <= timestamp:
            continue
        if scenes and timestamp <= scenes[-1]['end'] + 1e-8:
            scenes[-1]['end'] = max(scenes[-1]['end'], end)
            scenes[-1]['score'] = max(scenes[-1]['score'], score)
        else:
            scenes.append(dict(start=timestamp, end=end, score=score))
    return threshold, scenes


def refine_scenes(times, scores, duration, step, threshold, seed_stride, max_gap):
    """Expand coarse seeds over a finer grid; merge only verified weak gaps."""
    intervals = []
    covered = -1
    for seed in range(0, len(times), seed_stride):
        if seed <= covered or scores[seed] < threshold:
            continue
        left = right = seed
        while left > 0 and scores[left - 1] >= threshold:
            left -= 1
        while right + 1 < len(times) and scores[right + 1] >= threshold:
            right += 1
        intervals.append((left, right))
        covered = right

    # Relax toward lower similarity even when users choose a negative threshold.
    gap_threshold = threshold - abs(threshold) * .2
    merged = []
    for left, right in intervals:
        if merged:
            previous_left, previous_right = merged[-1]
            gap = times[left] - min(times[previous_right] + step, duration)
            if gap <= max_gap and all(score >= gap_threshold
                                      for score in scores[previous_right + 1:left]):
                merged[-1] = (previous_left, right)
                continue
        merged.append((left, right))
    return [dict(start=times[left], end=min(times[right] + step, duration),
                 score=max(scores[left:right + 1]),
                 peak_time=times[max(range(left,right + 1),key=scores.__getitem__)])
            for left, right in merged]


def sample_frames(path, interval):
    import cv2
    from PIL import Image
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('interval must be positive and finite')
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'cannot open video: {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('video has no valid FPS')
        step = max(1, round(interval * fps))
        index = 0
        while cap.grab():
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f'cannot decode frame {index}')
                yield index / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), step / fps
            index += 1
        return index / fps
    finally:
        cap.release()


class ClipEncoder:
    cache_id = 'openclip:xlm-roberta-base-ViT-B-32:laion5b_s13b_b90k'

    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'xlm-roberta-base-ViT-B-32', pretrained='laion5b_s13b_b90k', device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
        self._query = None
        self._text = None

    def images(self, frames):
        batch = self.torch.stack([self.preprocess(frame) for frame in frames]).to(self.device)
        with self.torch.inference_mode():
            return self.model.encode_image(batch, normalize=True).cpu()

    def scores(self, vectors, query):
        from query_support import validate_visual_query
        validate_visual_query(query)
        if query != self._query:
            tokens = self.tokenizer([query]).to(self.device)
            with self.torch.inference_mode():
                text = self.model.encode_text(tokens, normalize=True).cpu()
            self._query, self._text = query, text
        return (vectors @ self._text.T).flatten().tolist()


def index_video(path, interval, encoder, batch_size=16):
    times, vectors, batch = [], [], []
    frames = sample_frames(path, interval)
    while True:
        try:
            timestamp, frame, actual_interval = next(frames)
        except StopIteration as done:
            duration = done.value
            break
        times.append(timestamp)
        batch.append(frame)
        if len(batch) == batch_size:
            vectors.append(encoder.images(batch))
            batch = []
    if batch:
        vectors.append(encoder.images(batch))
    if not times:
        raise ValueError('video contains no decodable frames')
    return times, encoder.torch.cat(vectors), duration, actual_interval


def main():
    parser = argparse.ArgumentParser(description='Find approximate video scenes with Chinese or English text.')
    parser.add_argument('video')
    parser.add_argument('query', nargs='?', help='omit for interactive queries; :quit exits')
    parser.add_argument('--interval', type=positive, default=2.)
    parser.add_argument('--boundary-step', type=positive, default=.5,
                        help='finer grid for boundary search (default: 0.5 seconds)')
    parser.add_argument('--max-gap', type=float, default=5.,
                        help='maximum verified gap to merge (default: 5 seconds)')
    parser.add_argument('--min-score', type=float, default=.22)
    parser.add_argument('--sigma', type=float, default=2.)
    parser.add_argument('--threshold', type=float,
                        help='fixed cosine threshold; overrides the distribution threshold')
    args = parser.parse_args()
    if not math.isfinite(args.min_score) or not -1 <= args.min_score <= 1:
        parser.error('--min-score must be between -1 and 1')
    if not math.isfinite(args.sigma) or args.sigma < 0:
        parser.error('--sigma must be finite and nonnegative')
    if not math.isfinite(args.max_gap) or args.max_gap < 0:
        parser.error('--max-gap must be finite and nonnegative')
    if args.threshold is not None and (not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1):
        parser.error('--threshold must be between -1 and 1')
    if args.query is not None and not args.query.strip():
        parser.error('query cannot be blank')
    from pathlib import Path
    if not Path(args.video).is_file():
        parser.error('video file does not exist')
    try:
        encoder = ClipEncoder()
        print(f'Indexing on {encoder.device} ...', file=sys.stderr)
        times, vectors, duration, step = index_video(
            args.video, min(args.interval, args.boundary_step), encoder)
        seed_stride = max(1, round(args.interval / step))

        def run(query):
            scores = encoder.scores(vectors, query)
            threshold, _ = select_scenes(times[::seed_stride], scores[::seed_stride],
                                         duration, step * seed_stride, args.min_score, args.sigma)
            if args.threshold is not None:
                threshold = args.threshold
            scenes = refine_scenes(times, scores, duration, step, threshold,
                                   seed_stride, args.max_gap)
            print(json.dumps(dict(query=query, threshold=threshold,
                                  sample_interval=step * seed_stride,
                                  boundary_step=step, scenes=scenes), ensure_ascii=False))

        if args.query is not None:
            run(args.query)
        else:
            print('Enter Chinese or English text; :quit exits.', file=sys.stderr)
            for line in sys.stdin:
                query = line.strip()
                if query == ':quit':
                    break
                if query:
                    run(query)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
