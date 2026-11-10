"""Optional real-model smoke test. Downloads weights on first run."""
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

def main():
    root = Path(__file__).resolve().parent
    # This reproducible test uses the project's downloaded weights, even when
    # a machine-wide cache environment variable points somewhere else.
    os.environ['HF_HOME'] = str(root / '.cache' / 'huggingface')
    os.environ['TORCH_HOME'] = str(root / '.cache' / 'torch')
    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / 'colors.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (224, 224))
        if not writer.isOpened():
            raise RuntimeError('cannot create smoke-test video')
        try:
            for color, count in [((0, 0, 255), 10), ((255, 0, 0), 20), ((0, 0, 255), 10)]:
                for _ in range(count):
                    writer.write(np.full((224, 224, 3), color, dtype=np.uint8))
        finally:
            writer.release()
        command = [sys.executable, str(root / 'search.py'), str(video),
                   '--interval', '2', '--boundary-step', '.5', '--min-score', '-1', '--sigma', '0']
        once = subprocess.run(command + ['a blue screen'], capture_output=True,
                              text=True, check=True, timeout=180)
        result = json.loads(once.stdout)
        expected = [(1., 3.)]
        if [(scene['start'], scene['end']) for scene in result['scenes']] != expected:
            raise RuntimeError(f'incorrect boundaries: {result}')
        if not all(math.isfinite(scene['score']) for scene in result['scenes']):
            raise RuntimeError('invalid model scores')
        interactive = subprocess.run(command, input='a blue screen\na red screen\n纯蓝色画面\n:quit\n',
                                     capture_output=True, encoding='utf-8',
                                     env=dict(os.environ, PYTHONIOENCODING='utf-8'), check=True, timeout=180)
        results = [json.loads(line) for line in interactive.stdout.splitlines()]
        if len(results) != 3 or results[0] != result or results[1]['query'] != 'a red screen':
            raise RuntimeError('interactive CLI output differs from one-shot search')
        if [(s['start'],s['end']) for s in results[2]['scenes']] != expected:
            raise RuntimeError('generic Chinese query failed without any reference profile')
        if interactive.stderr.count('Indexing on') != 1:
            raise RuntimeError('interactive session indexed more than once')
        dense = subprocess.run(
            [sys.executable, str(root / 'search.py'), str(video), 'a red screen',
             '--interval', '.5', '--max-gap', '0', '--min-score', '-1', '--sigma', '0'],
            capture_output=True, text=True, check=True, timeout=180)
        dense_result = json.loads(dense.stdout)
        if [(s['start'], s['end']) for s in dense_result['scenes']] != [(0., 1.), (3., 4.)]:
            raise RuntimeError(f'dense search failed to preserve the red segments: {dense_result}')
        print(json.dumps(dict(one_shot=result, interactive=results, dense_red=dense_result), indent=2))


if __name__ == '__main__':
    main()
