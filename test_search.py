import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import search
from search import index_video, positive, sample_frames, select_scenes


class SceneTests(unittest.TestCase):
    def test_low_scores_have_no_hits(self):
        threshold, scenes = select_scenes([0, 2], [.1, .1], 4, 2)
        self.assertEqual(threshold, .22)
        self.assertEqual(scenes, [])

    def test_population_deviation(self):
        threshold, _ = select_scenes([0, 2], [.2, .4], 4, 2, sigma=1)
        self.assertAlmostEqual(threshold, .4)

    def test_merge_neighbors_keep_gap_clip_tail(self):
        _, scenes = select_scenes([0, 2, 4, 6], [.8, .9, .1, .8], 7, 2, .5, 0)
        self.assertEqual([(s['start'], s['end']) for s in scenes], [(0, 4), (6, 7)])
        self.assertEqual(scenes[0]['score'], .9)

    def test_short_video(self):
        _, scenes = select_scenes([0], [.8], .3, 2)
        self.assertEqual(scenes[0]['end'], .3)

    def test_invalid_intervals(self):
        for value in ['0', '-2', 'nan', 'inf']:
            with self.assertRaises(argparse.ArgumentTypeError):
                positive(value)

    def test_mismatched_input(self):
        with self.assertRaises(ValueError):
            select_scenes([0], [], 2, 2)


class RefinementTests(unittest.TestCase):
    def refine(self, scores, stride=4, duration=None, max_gap=5):
        times = [i * .5 for i in range(len(scores))]
        return search.refine_scenes(times, scores, duration or len(scores) * .5,
                                    .5, .6, stride, max_gap)

    def test_expands_beyond_seed_and_stops_at_unrelated_frames(self):
        scenes = self.refine([.1, .1, .7, .8, .9, .7, .1, .1, .1])
        self.assertEqual([(s['start'], s['end']) for s in scenes], [(1, 3)])
        self.assertEqual(scenes[0]['peak_time'],2)

    def test_peak_uses_best_sample_across_merged_intervals(self):
        scenes=self.refine([.8,.5,.5,.5,.9])
        self.assertEqual(scenes[0]['peak_time'],2)
        self.assertEqual(scenes[0]['score'],.9)

    def test_merges_gap_only_when_all_gap_samples_pass(self):
        scenes = self.refine([.8, .5, .5, .5, .9])
        self.assertEqual([(s['start'], s['end']) for s in scenes], [(0, 2.5)])
        scenes = self.refine([.8, .5, .1, .5, .9])
        self.assertEqual([(s['start'], s['end']) for s in scenes], [(0, .5), (2, 2.5)])

    def test_long_gaps_are_not_merged(self):
        scenes = self.refine([.8, .5, .5, .5, .9], max_gap=1)
        self.assertEqual(len(scenes), 2)

    def test_clips_final_cell_to_duration(self):
        scenes = self.refine([.8, .8, .8], stride=1, duration=1.2)
        self.assertEqual([(s['start'], s['end']) for s in scenes], [(0, 1.2)])

    def test_no_coarse_seed_means_no_scene(self):
        self.assertEqual(self.refine([.1, .9, .1, .1, .1]), [])

    def test_negative_threshold_is_relaxed_for_gap(self):
        scenes = search.refine_scenes([0, .5, 1], [-.2, -.32, -.2],
                                      1.5, .5, -.3, 2, 5)
        self.assertEqual(len(scenes), 1)


class CommandTests(unittest.TestCase):
    def test_fixed_threshold_can_recover_uniformly_relevant_video(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as video:
            with patch.object(search, 'ClipEncoder') as encoder_class, \
                    patch.object(search, 'index_video') as index, \
                    patch('sys.argv', ['search.py', video.name, 'snow', '--threshold', '.25']), \
                    patch('sys.stderr', new_callable=io.StringIO), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                index.return_value = ([0, .5, 1, 1.5], object(), 2, .5)
                encoder_class.return_value.scores.return_value = [.35, .36, .37, .36]
                self.assertEqual(search.main(), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result['threshold'], .25)
                self.assertEqual([(s['start'], s['end']) for s in result['scenes']], [(0, 2)])

    def test_invalid_fixed_threshold_fails_before_loading_model(self):
        for value in ['nan', 'inf', '1.1', '-1.1']:
            with patch('sys.argv', ['search.py', 'unused', '--threshold', value]), \
                    patch('sys.stderr', new_callable=io.StringIO), \
                    patch.object(search, 'ClipEncoder') as encoder_class:
                with self.assertRaises(SystemExit) as error:
                    search.main()
                self.assertEqual(error.exception.code, 2)
                encoder_class.assert_not_called()

    def test_interactive_queries_reuse_model_and_index(self):
        with tempfile.NamedTemporaryFile(suffix='.avi') as video:
            with patch.object(search, 'ClipEncoder') as encoder_class, \
                    patch.object(search, 'index_video') as index, \
                    patch('sys.argv', ['search.py', video.name, '--sigma', '0']), \
                    patch('sys.stdin', io.StringIO('blue\nred\n:quit\nignored\n')), \
                    patch('sys.stderr', new_callable=io.StringIO), \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                index.return_value = ([0, .5, 1, 1.5], object(), 2, .5)
                encoder_class.return_value.device = 'test'
                encoder_class.return_value.scores.side_effect = [[.8] * 4, [.7] * 4]
                self.assertEqual(search.main(), 0)
                self.assertEqual(encoder_class.call_count, 1)
                self.assertEqual(index.call_count, 1)
                results = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([result['query'] for result in results], ['blue', 'red'])


class VideoTests(unittest.TestCase):
    def test_decode_batch_and_select(self):
        import cv2
        import numpy as np

        class TensorOps:
            cat = staticmethod(np.concatenate)

        class Encoder:
            torch = TensorOps()
            def images(self, frames):
                return np.array([[np.asarray(frame).mean()] for frame in frames])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(30):
                writer.write(np.full((32, 32, 3), 0 if i < 10 else 255, dtype=np.uint8))
            writer.release()
            times, vectors, duration, interval = index_video(path, .5, Encoder(), batch_size=2)
            self.assertEqual(times, [0, .5, 1, 1.5, 2, 2.5])
            self.assertEqual(vectors.shape, (6, 1))
            self.assertAlmostEqual(duration, 3)
            _, scenes = select_scenes(times, (vectors[:, 0] / 255).tolist(), duration, interval, .5, 0)
            self.assertEqual([(s['start'], s['end']) for s in scenes], [(1, 3)])

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            next(sample_frames('nonexistent-video.avi', 2))


if __name__ == '__main__':
    unittest.main()
