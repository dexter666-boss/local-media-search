import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from speech_search import SpeechSearch,find_matches
from library import Library


def segment(text,start=1,end=3):return dict(text=text,start=start,end=end)


class SpeechTests(unittest.TestCase):
    def test_simplified_query_matches_traditional_transcript_and_preserves_original(self):
        rows=[segment('中國市場發展迅速。',2,5)]
        result=find_matches(rows,'中国市场')
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['text'],'中國市場發展迅速。')
        self.assertEqual((result[0]['start'],result[0]['end']),(2,5))
        self.assertEqual(rows[0]['text'],'中國市場發展迅速。')

    def test_traditional_query_and_cross_segment_matches(self):
        rows=[segment('中国',2,3),segment('市场发展',3,6)]
        result=find_matches(rows,'中國市場')
        self.assertEqual((result[0]['start'],result[0]['end']),(2,6))
        self.assertEqual(result[0]['text'],'中国 市场发展')

    def test_literal_hits_survive_phrase_context_and_duplicate_hits_merge(self):
        self.assertTrue(find_matches([segment('乾隆')],'乾'))
        self.assertEqual(len(find_matches([segment('中国市场')],'中国')),1)
        self.assertFalse(find_matches([segment('頭髮')],'发票'))

    def test_simplification_does_not_expand_into_number_or_fuzzy_matching(self):
        self.assertFalse(find_matches([segment('一百二米')],'120米'))
        self.assertFalse(find_matches([segment('明天開會')],'明天聚会'))
        self.assertFalse(find_matches([segment('中國',1,2),segment('市場',8,9)],'中国市场'))

    def test_quantity_variants_preserve_original_and_time(self):
        rows=[segment('從一百米到四百米，三点开会。',2,8)]
        result=find_matches(rows,'从100米到400米')
        self.assertEqual(result[0]['text'],rows[0]['text'])
        self.assertEqual((result[0]['start'],result[0]['end']),(2,8))
        self.assertTrue(find_matches(rows,'3点开会'))
        self.assertTrue(find_matches([segment('销售52亿元')],'销售五十二亿元'))
        self.assertTrue(find_matches([segment('二〇二四年三月四日')],'2024年3月4日'))

    def test_quantity_guards_avoid_ambiguous_numbers_and_plain_words(self):
        for original,query in [('一心一意','1心1意'),('万一天气变坏','10001天气变坏'),
                               ('第003号','第3号'),('二三十人','230人'),('一百二米','120米'),
                               ('三点五米','35米'),('一点也不','1点也不'),('四分之一米','1米'),('负三米','3米')]:
            self.assertFalse(find_matches([segment(original)],query),(original,query))
        self.assertTrue(find_matches([segment('第003号')],'003'))
        self.assertTrue(find_matches([segment('三点五米')],'三点五米'))

    def test_quantities_do_not_change_numeric_values_or_gap_rules(self):
        self.assertFalse(find_matches([segment('三点开会')],'2点开会'))
        self.assertFalse(find_matches([segment('一百米')],'10'))
        self.assertFalse(find_matches([segment('100米')],'一百'))
        self.assertFalse(find_matches([segment('一百米',1,2),segment('跑完',8,9)],'100米跑完'))
        self.assertEqual(len(find_matches([segment('100米')],'100米')),1)

    def test_literal_normalization_and_cross_segment_timing(self):
        rows=[segment('明天下午，',2,4),segment('三点开会。',4.3,6)]
        result=find_matches(rows,'明天 下午三点开会')
        self.assertEqual((result[0]['start'],result[0]['end']),(2,6))
        self.assertTrue(find_matches([segment('Project ALPHA!')],'project alpha'))
        self.assertFalse(find_matches(rows,'约时间见面'))

    def test_long_gaps_do_not_form_fake_sentence_and_repeated_hits_deduplicate(self):
        self.assertEqual(find_matches([segment('明天',1,2),segment('开会',10,12)],'明天开会'),[])
        self.assertEqual(len(find_matches([segment('开会开会')],'开会')),1)
        with self.assertRaises(ValueError):find_matches([],' ， ')

    def test_cache_reuses_and_invalidates_file_and_model_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'a.wav';path.write_bytes(b'fixture')
            search=SpeechSearch();search.transcribe=Mock(return_value=dict(segments=[],status='no_speech'))
            cache=Path(temp)/'cache'
            self.assertFalse(search.document(path,cache)[1])
            self.assertTrue(search.document(path,cache)[1])
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            self.assertFalse(search.document(path,cache)[1])
            search.cache_id+='new'
            self.assertFalse(search.document(path,cache)[1])
            self.assertEqual(search.transcribe.call_count,3)
            next(cache.glob('*.json')).write_text('{invalid')

    def test_real_audio_decoder_compatibility_without_model_download(self):
        import wave
        from faster_whisper.audio import decode_audio
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'silence.wav'
            with wave.open(str(path),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(bytes(16000))
            decoded=decode_audio(str(path))
            self.assertEqual(len(decoded),8000)
            self.assertEqual(float(abs(decoded).max()),0.)

    def test_transcription_clips_timestamps_to_duration_and_drops_empty_ranges(self):
        import wave
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'speech.wav'
            with wave.open(str(path),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
                stream.writeframes(bytes(115370))
            duration=3.6053333333333333
            rows=[SimpleNamespace(start=.27,end=4.27,text=' Dogs are sitting by the door! '),
                  SimpleNamespace(start=-1,end=.2,text='beginning'),
                  SimpleNamespace(start=4,end=5,text='outside'),
                  SimpleNamespace(start=1,end=1,text='empty'),
                  SimpleNamespace(start=1,end=2,text=' ')]
            search=SpeechSearch();search.model=Mock()
            search.model.transcribe.return_value=(iter(rows),SimpleNamespace(duration=duration,language='en'))
            result=search.transcribe(path)
            self.assertEqual(result['status'],'transcribed')
            self.assertEqual(result['segments'],[
                dict(start=.27,end=duration,text='Dogs are sitting by the door!'),
                dict(start=0.,end=.2,text='beginning')])
            self.assertEqual(find_matches(result['segments'],'Dogs')[0]['end'],duration)
            search.model.transcribe.return_value=(iter(rows[2:]),SimpleNamespace(duration=duration,language='en'))
            self.assertEqual(search.transcribe(path)['status'],'no_speech')

    def test_missing_audio_stream_does_not_load_model(self):
        import cv2,numpy as np
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'silent.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
            self.assertTrue(writer.isOpened());writer.write(np.zeros((32,32,3),dtype=np.uint8));writer.release()
            search=SpeechSearch()
            self.assertEqual(search.transcribe(path)['status'],'no_audio')
            self.assertIsNone(search.model)

    def test_audio_only_registration_and_visual_scan_excludes_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ['speech.wav','speech.mp3','clip.mp4','image.jpg','ignore.txt']:(folder/name).write_bytes(b'fixture')
            library=Library(folder/'cache')
            result=library.scan(folder,None,speech_only=True)
            self.assertEqual((result['audio'],result['videos'],result['photos']),(2,1,0))
            self.assertEqual(set(item['kind'] for item in library.items.values()),{'audio','video'})
            library.scan(folder,None)
            self.assertFalse(any(item['kind']=='audio' for item in library.items.values()))

    def test_counts_precede_limits_and_audio_avoids_visual_thumbnail(self):
        search=SpeechSearch();search.document=Mock(return_value=(dict(segments=[segment('开会')],status='transcribed'),True))
        items={str(i):dict(path=Path(f'{i}.wav'),name=f'{i}.wav',kind='audio') for i in range(55)}
        with patch('library.thumbnail',side_effect=AssertionError('audio has no image')):
            result=search.search(items,'开会','unused')
        self.assertEqual(result['total_files'],55);self.assertEqual(len(result['files']),50)
        self.assertEqual(result['files'][0]['segments'][0]['text'],'开会')

    def test_no_speech_no_audio_errors_and_progress_remain_distinct(self):
        search=SpeechSearch()
        search.document=Mock(side_effect=[(dict(segments=[],status='no_audio'),True),
                                         (dict(segments=[],status='no_speech'),False),ValueError('bad file')])
        items={str(i):dict(path=Path(f'{i}.mp4'),name=f'{i}.mp4',kind='video') for i in range(3)}
        events=[];result=search.search(items,'开会','unused',progress=lambda **state:events.append(state))
        self.assertEqual((result['no_audio'],result['no_speech'],len(result['errors'])),(1,1,1))
        self.assertEqual(events[-1]['done'],2)
        self.assertEqual(result['total_files'],0)


if __name__=='__main__':unittest.main()
