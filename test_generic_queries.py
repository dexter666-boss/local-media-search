from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from library import Library
from search import ClipEncoder


class GenericQueryTests(unittest.TestCase):
    def test_folder_query_encodes_text_once_and_changed_query_recomputes(self):
        import torch
        encoder=ClipEncoder.__new__(ClipEncoder)
        encoder.torch=torch
        encoder.device='cpu'
        encoder._query=None
        encoder._text=None
        encoder.tokenizer=Mock(return_value=torch.tensor([[1]]))
        encoder.model=Mock()
        encoder.model.encode_text.side_effect=[torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])]
        vectors=torch.tensor([[1.,0.],[0.,1.]])
        for _ in range(10):
            self.assertEqual(encoder.scores(vectors,'雪山'),[1.,0.])
        self.assertEqual(encoder.model.encode_text.call_count,1)
        self.assertEqual(encoder.scores(vectors,'海洋'),[0.,1.])
        self.assertEqual(encoder.model.encode_text.call_count,2)
        self.assertEqual(encoder.tokenizer.call_count,2)

    def test_named_and_unseen_queries_use_the_same_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            library=Library(folder)
            library.items={'item':dict(name='sample.mp4',path=Path('sample.mp4'),kind='video',
                                      index=([0.],object(),1.,.5))}
            encoder=Mock()
            encoder.scores.return_value=[.9]
            with patch('library.thumbnail',return_value=None):
                for query in ['日落之城击杀视频','未知游戏的新地图','雪地里的孩子','a child on snow']:
                    result=library.search(query,.25,encoder)
                    self.assertEqual(result['total_files'],1)
                    encoder.scores.assert_called_with(library.items['item']['index'][1],query)
                    self.assertNotIn('method',result)

    def test_default_code_has_no_game_profile_dependency(self):
        root=Path(__file__).resolve().parent
        for name in ['app.py','library.py','search.py','query_support.py','ui.html']:
            source=(root/name).read_text(encoding='utf-8')
            self.assertNotIn('from valorant',source)
            self.assertNotIn('SunsetFilter',source)
            self.assertNotIn('日落之城',source)


if __name__ == '__main__':
    unittest.main()
