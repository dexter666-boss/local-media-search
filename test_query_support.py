import tempfile
import unittest
from unittest.mock import Mock

from app import Session
from library import Library
from query_support import validate_visual_query


class QuerySupportTests(unittest.TestCase):
    def test_explicit_audio_queries_are_not_scored(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Library(folder)
            encoder = Mock()
            for query in ['找交换余生这首歌', 'background music by JJ Lin', 'a song called Exchange']:
                with self.assertRaises(ValueError):
                    library.search(query,.25,encoder)
            encoder.scores.assert_not_called()

    def test_single_video_rejects_song_before_scoring(self):
        session = Session()
        session.encoder = Mock()
        with self.assertRaises(ValueError):
            session.search('找交换余生这首歌',.25)
        session.encoder.scores.assert_not_called()

    def test_supported_visual_queries_remain_available(self):
        for query in ['a child snowboarding','a singer on stage','日落之城击杀视频','雪地里的孩子','未知游戏地图上的人物']:
            validate_visual_query(query)
        with self.assertRaises(ValueError):
            validate_visual_query('Exchange',target='music')


if __name__ == '__main__':
    unittest.main()
