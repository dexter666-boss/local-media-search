import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from library import Library
from text_search import TextSearch, matching_lines


def line(text):
    return dict(text=text,confidence=.99,box=[[.1,.1],[.9,.1],[.9,.2],[.1,.2]])


class TextSearchTests(unittest.TestCase):
    def test_wraps_width_case_and_whitespace_but_preserves_punctuation(self):
        lines=[line('明天下午'),line('三点开会'),line('ＡＢＣ 123！')]
        self.assertEqual(matching_lines(lines,'明天 下午三点开会'),lines[:2])
        self.assertEqual(matching_lines(lines,'abc123!'),lines[2:])
        self.assertEqual(matching_lines(lines,'ABC123?'),[])
        self.assertEqual(matching_lines(lines,'明天见面'),[])

    def test_repeated_matches_return_each_line_once_and_blank_is_invalid(self):
        lines=[line('开会开会'),line('开会')]
        self.assertEqual(matching_lines(lines,'开会'),lines)
        with self.assertRaises(ValueError):
            matching_lines(lines,'  ')

    def test_cache_reuses_across_queries_and_instances_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            path=folder/'sample.png'
            Image.new('RGB',(40,40)).save(path)
            search=TextSearch()
            search.recognize=Mock(return_value=dict(size=[40,40],lines=[line('开会')]))
            self.assertFalse(search.document(path,folder/'ocr')[1])
            other=TextSearch()
            other.recognize=Mock(side_effect=AssertionError('Cache should be reused'))
            self.assertTrue(other.document(path,folder/'ocr')[1])
            stat=path.stat()
            os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            self.assertFalse(search.document(path,folder/'ocr')[1])
            self.assertEqual(search.recognize.call_count,2)
            search.cache_id+=':new-version'
            self.assertFalse(search.document(path,folder/'ocr')[1])

    def test_broken_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            path=folder/'sample.png'
            path.write_bytes(b'fixture')
            search=TextSearch()
            search.recognize=Mock(return_value=dict(size=[10,10],lines=[]))
            search.document(path,folder/'ocr')
            next((folder/'ocr').glob('*.json')).write_text('{broken')
            self.assertFalse(search.document(path,folder/'ocr')[1])

    def test_text_scan_needs_no_encoder_and_visual_search_requests_rescan(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            Image.new('RGB',(30,30)).save(folder/'a.png')
            library=Library(folder/'cache')
            result=library.scan(folder,None)
            self.assertEqual(result['photos'],1)
            self.assertIsNone(next(iter(library.items.values()))['index'])
            with self.assertRaisesRegex(ValueError,'重新扫描'):
                library.search('海边',.2,None)

    def test_failed_images_and_videos_are_reported_not_misreported_as_no_text(self):
        search=TextSearch()
        search.document=Mock(side_effect=[(dict(size=[10,10],lines=[line('<script>开会</script>')]),True),OSError('broken')])
        items={'a':dict(path=Path('a.png'),name='a.png',kind='image'),
               'b':dict(path=Path('b.png'),name='b.png',kind='image'),
               'c':dict(path=Path('c.mp4'),name='c.mp4',kind='video')}
        with patch('library.thumbnail',return_value=None):
            result=search.search(items,'开会','unused')
        self.assertEqual(result['total_files'],1)
        self.assertEqual(result['ocr_processed'],1)
        self.assertEqual(result['ocr_reused'],1)
        self.assertEqual(result['skipped_videos'],1)
        self.assertEqual(len(result['errors']),1)
        self.assertEqual(result['files'][0]['ocr_lines'][0]['text'],'<script>开会</script>')

    def test_ocr_progress_counts_images_errors_and_cache_but_skips_videos(self):
        search=TextSearch()
        search.document=Mock(side_effect=[(dict(size=[10,10],lines=[]),True),OSError('broken')])
        items={'a':dict(path=Path('a.png'),name='a.png',kind='image'),
               'b':dict(path=Path('b.png'),name='b.png',kind='image'),
               'c':dict(path=Path('c.mp4'),name='c.mp4',kind='video')}
        events=[]
        result=search.search(items,'开会','unused',progress=lambda **state:events.append(state))
        self.assertEqual(result['skipped_videos'],1)
        self.assertEqual((events[-1]['done'],events[-1]['total'],events[-1]['errors'],events[-1]['reused']),(2,2,1,1))
        self.assertEqual(events[0]['current'],'a.png')

    def test_no_filename_matching_and_total_count_precedes_display_limit(self):
        search=TextSearch()
        search.document=Mock(return_value=(dict(size=[10,10],lines=[line('会议通知')]),True))
        items={str(i):dict(path=Path(f'{i}.png'),name=f'{i}.png',kind='image') for i in range(55)}
        with patch('library.thumbnail',return_value=None):
            result=search.search(items,'会议','unused')
            self.assertEqual(result['total_files'],55)
            self.assertEqual(len(result['files']),50)
            result=search.search(items,'0.png','unused')
            self.assertEqual(result['total_files'],0)

    def test_exif_orientation_and_color_conversion_preserve_overlay_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rotate.jpg'
            image=Image.new('RGB',(80,40),(255,0,0))
            exif=Image.Exif();exif[274]=6
            image.save(path,exif=exif)
            search=TextSearch()
            search.engine=Mock(return_value=([([[0,0],[40,0],[40,80],[0,80]],'开会',.9)],None))
            document=search.recognize(path)
            self.assertEqual(document['size'],[40,80])
            self.assertEqual(document['lines'][0]['box'][2],[1.,1.])
            array=search.engine.call_args.args[0]
            self.assertEqual(array.shape,(80,40,3))
            self.assertGreater(int(array[0,0,2]),240)


if __name__=='__main__':
    unittest.main()
