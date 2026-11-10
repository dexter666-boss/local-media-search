from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from condition_search import parse_conditions,search_conditions


class ConditionSearchTests(unittest.TestCase):
    def test_parse_simple_positive_context_and_object(self):
        for text,context,category in [('雪地里的狗','雪地','狗'),('在草地上面的猫','草地','猫'),('房间里面的狗','房间','狗')]:
            self.assertEqual(parse_conditions(text),dict(context=context,object=category))

    def test_unsupported_constraints_do_not_get_silently_discarded(self):
        for text in ('没有雪的草地上的狗','雪地里的狗和猫','雪地里的黑色狗','沙发上睡觉的狗',
                     '沙发上的猫','汽车里的狗','狗旁边的猫','树上的鸟','狗','a dog on snow'):
            self.assertIsNone(parse_conditions(text),text)

    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.root=Path(self.folder.name)
        self.encoder=SimpleNamespace(torch=SimpleNamespace(cat=lambda rows:np.concatenate(rows)))
        self.encoder.scores=Mock(side_effect=lambda vectors,query:vectors[:,0].tolist())
        self.detector=SimpleNamespace(default_threshold=.8,evidence=Mock())

    def tearDown(self):
        self.folder.cleanup()

    def item(self,name,score):
        path=self.root/(name+'.png')
        Image.new('RGB',(16,16)).save(path)
        return dict(path=path,name=name,kind='image',index=([0.],np.array([[score]]),0.,.5))

    def test_both_conditions_required_and_scores_stay_separate(self):
        items={'snow_dog':self.item('snow_dog',.3),'grass_dog':self.item('grass_dog',.1),'snow_cat':self.item('snow_cat',.4)}
        self.encoder.scores.side_effect=lambda vectors,query:(vectors[:,0]+(0 if query=='雪地' else .4)).tolist()
        self.detector.evidence.return_value=[dict(file_id=key,name=key,kind='image',start=None,end=None,
            score=.95 if key.endswith('dog') else .01) for key in items]
        result=search_conditions(items,'雪地里的狗',dict(context='雪地',object='狗'),self.encoder,self.detector,self.root,.2)
        self.assertEqual(set(self.detector.evidence.call_args.args[0]),set(items))
        self.assertEqual(result['total_files'],1)
        self.assertEqual(result['files'][0]['conditions'][0]['score'],.95)
        self.assertEqual(result['files'][0]['conditions'][1]['score'],.3)
        self.assertAlmostEqual(result['files'][0]['score'],.7)
        self.assertEqual(self.encoder.scores.call_count,2)
        self.assertEqual(result['review_total_files'],2)
        self.assertEqual({r['file_id'] for r in result['files']+result['review_files']},set(items))

    def test_no_context_candidates_skips_detector_and_videos_are_rejected(self):
        items={'photo':self.item('photo',.1)}
        result=search_conditions(items,'雪地里的狗',dict(context='雪地',object='狗'),self.encoder,self.detector,self.root,.2)
        self.assertEqual(result['total_files'],0)
        self.detector.evidence.assert_not_called()
        items['photo']['kind']='video'
        with self.assertRaises(ValueError):
            search_conditions(items,'雪地里的狗',dict(context='雪地',object='狗'),self.encoder,self.detector,self.root,.2)

    def test_full_candidates_are_filtered_before_display_limit(self):
        items={str(i):self.item(str(i),.3) for i in range(55)}
        self.detector.evidence.return_value=[dict(file_id=key,name=key,kind='image',start=None,end=None,score=.99) for key in items]
        result=search_conditions(items,'雪地里的狗',dict(context='雪地',object='狗'),self.encoder,self.detector,self.root,.2)
        self.assertEqual(len(self.detector.evidence.call_args.args[0]),55)
        self.assertEqual(result['total_files'],55)
        self.assertEqual(len(result['files']),50)

    def test_failed_object_condition_is_retained_separately_even_without_primary_results(self):
        items={str(i):self.item(str(i),.3) for i in range(55)}
        self.detector.evidence.return_value=[dict(file_id=key,name=key,kind='image',start=None,end=None,score=0.) for key in items]
        result=search_conditions(items,'雪地里的猫',dict(context='雪地',object='猫'),self.encoder,self.detector,self.root,.2)
        self.assertEqual(result['total_files'],0)
        self.assertEqual(result['files'],[])
        self.assertEqual(result['review_total_files'],55)
        self.assertEqual(len(result['review_files']),50)
        self.assertFalse(result['review_files'][0]['conditions'][0]['passed'])
        self.assertTrue(result['review_files'][0]['conditions'][1]['passed'])


if __name__=='__main__': unittest.main()
