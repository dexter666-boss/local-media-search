from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image
import torch
from object_search import ObjectSearch


class ObjectSearchTests(unittest.TestCase):
    def test_routing_supports_exact_standard_names_not_partial_or_compound_words(self):
        for query in ('猫',' 狗 ','Cat','person','hot dog','热狗'):
            self.assertTrue(ObjectSearch.supports(query),query)
        for query in ('雪地里的狗','没有猫','猫狗','无人机','儿童','篮球','catfish','dog in snow','__background__','N/A',''):
            self.assertFalse(ObjectSearch.supports(query),query)

    def test_model_profiles_have_separate_recommended_thresholds_and_cache_ids(self):
        self.assertEqual(ObjectSearch.threshold_for('fast'),.5)
        self.assertEqual(ObjectSearch.threshold_for('accurate'),.8)
        self.assertNotEqual(ObjectSearch.model_id,ObjectSearch.accurate_model_id)
        with self.assertRaises(ValueError):
            ObjectSearch.threshold_for('unknown')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.path=self.root/'photo.png'
        Image.new('RGB',(32,32),'white').save(self.path)
        self.items={'photo':dict(path=self.path,name='photo.png',kind='image')}
        self.search=ObjectSearch.__new__(ObjectSearch)
        self.search.names={'人':1,'猫':17}
        self.search.torch=torch
        self.search.device='cpu'
        self.search.transform=Mock(return_value=torch.zeros(3,32,32))
        self.search.model=Mock(return_value=[dict(labels=torch.tensor([1,1,17]),scores=torch.tensor([.8,.6,.3]))])

    def tearDown(self):
        self.temp.cleanup()

    def test_cache_reuses_all_classes_without_mixing_queries_or_thresholds(self):
        search=self.search
        first=search.search(self.items,'人',.5,self.root/'cache')
        self.assertEqual(first['total_files'],1)
        self.assertAlmostEqual(first['files'][0]['score'],.8)
        self.assertEqual(search.search(self.items,'猫',.5,self.root/'cache')['total'],0)
        self.assertEqual(search.search(self.items,'猫',.2,self.root/'cache')['total'],1)
        search.model.assert_called_once()

    def test_changed_file_is_redetected(self):
        self.search.search(self.items,'人',.5,self.root/'cache')
        Image.new('RGB',(64,64),'black').save(self.path)
        self.search.search(self.items,'人',.5,self.root/'cache')
        self.assertEqual(self.search.model.call_count,2)

    def test_changing_model_does_not_reuse_old_predictions(self):
        cache=self.root/'cache'
        self.assertEqual(self.search.search(self.items,'人',.5,cache)['total'],1)
        self.search.model_id=ObjectSearch.accurate_model_id
        self.search.model.return_value=[dict(labels=torch.tensor([1]),scores=torch.tensor([.1]))]
        self.assertEqual(self.search.search(self.items,'人',.5,cache)['total'],0)
        self.assertEqual(self.search.model.call_count,2)

    def test_unsupported_query_or_video_does_not_run_detector(self):
        with self.assertRaises(ValueError):
            self.search.search(self.items,'人正在跳舞',.5,self.root/'cache')
        self.items['photo']['kind']='video'
        with self.assertRaises(ValueError):
            self.search.search(self.items,'人',.5,self.root/'cache')
        self.search.model.assert_not_called()

    def test_accurate_dog_candidates_use_verification_before_returning(self):
        self.search.names['狗']=18
        self.search.verify_dogs=True
        self.search.dog_verifier=Mock()
        self.search.model.return_value=[dict(labels=torch.tensor([18]),scores=torch.tensor([.95]))]
        self.search.dog_verifier.score.return_value=0.
        cache=self.root/'cache'
        self.assertEqual(self.search.search(self.items,'狗',.8,cache)['total_files'],0)
        self.search.dog_verifier.score.return_value=.85
        self.assertEqual(self.search.search(self.items,'狗',.8,cache)['total_files'],1)
        self.search.model.assert_called_once()


if __name__=='__main__':
    unittest.main()
