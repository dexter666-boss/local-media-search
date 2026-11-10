import http.client
from pathlib import Path
import tempfile
import threading
import unittest
import json
from unittest.mock import patch

import cv2
import numpy as np
from app import Handler, Session, ThreadingHTTPServer, OBJECT_THRESHOLD
from library import Library, file_results
from PIL import Image


class Encoder:
    cache_id = 'test-brightness-v1'
    class torch:
        cat = staticmethod(np.concatenate)
        from_numpy = staticmethod(np.asarray)

    def images(self, frames):
        return np.array([[np.asarray(frame).mean() / 255] for frame in frames])

    def scores(self, vectors, query):
        return vectors[:, 0].tolist() if query == 'white screen' else [0.] * len(vectors)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.uploads = self.folder.name
        self.server.session = Session()
        self.server.session.encoder = Encoder()
        self.server.session.library = Library(Path(self.folder.name) / 'cache')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        video = Path(self.folder.name) / 'input.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
        self.assertTrue(writer.isOpened())
        for i in range(20):
            writer.write(np.full((32, 32, 3), 0 if i < 10 else 255, dtype=np.uint8))
        writer.release()
        self.video = video.read_bytes()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.folder.cleanup()

    def request(self, path, body, origin=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        headers = {'Content-Type': 'application/octet-stream'}
        if origin:
            headers['Origin'] = origin
        connection.request('POST', path, body, headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_text_only_scan_bypasses_visual_encoder_and_routes_literal_queries(self):
        folder=Path(self.folder.name)/'screenshots'
        folder.mkdir()
        Image.new('RGB',(40,40)).save(folder/'chat.png')
        self.server.session.encoder=None
        with patch('app.ClipEncoder',side_effect=AssertionError('Text scan must not load CLIP')):
            status,result=self.request('/api/library',json.dumps(dict(path=str(folder),target='text')))
        self.assertEqual(status,200)
        self.assertEqual(result['indexing'],'text')
        self.assertEqual(result['photos'],1)
        engine=unittest.mock.Mock()
        engine.search.return_value=dict(files=[],total_files=0,search_method='text')
        self.server.session.text_search=engine
        for target,query in [('text','交换余生'),('auto','文字：交换余生'),('auto','文字:交换余生')]:
            status,result=self.request('/api/library/search',json.dumps(dict(query=query,target=target)))
            self.assertEqual(status,200)
            self.assertEqual(result['search_method'],'text')
            self.assertEqual(engine.search.call_args.args[1],'交换余生')
        status,result=self.request('/api/library/search',json.dumps(dict(query='海边',target='visual')))
        self.assertEqual(status,400)
        self.assertIn('重新扫描',result['error'])

    def test_text_rejects_single_video_empty_query_and_threshold(self):
        status,_=self.request('/api/search',json.dumps(dict(query='开会',target='text')))
        self.assertEqual(status,400)
        self.server.session.library.items={'a':dict(path=Path('a.png'),kind='image',index=None)}
        for body in [dict(query='  ',target='text'),dict(query='文字：',target='auto'),dict(query='开会',target='text',threshold=.2)]:
            status,_=self.request('/api/library/search',json.dumps(body))
            self.assertEqual(status,400)

    def get_progress(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
        connection.request('GET','/api/progress')
        response=connection.getresponse()
        self.assertEqual(response.status,200)
        result=json.loads(response.read())
        connection.close()
        return result

    def test_progress_is_readable_while_scan_holds_task_lock(self):
        entered,release=threading.Event(),threading.Event()
        replies=[]
        def slow_scan(folder,encoder,progress):
            progress(phase='建立画面索引',done=1,total=3,current='sub/image.png',reused=1)
            entered.set()
            release.wait(5)
            return dict(files=0,photos=0,videos=0,reused=1,errors=[])
        with patch.object(self.server.session.library,'scan',side_effect=slow_scan):
            worker=threading.Thread(target=lambda:replies.append(self.request('/api/library',json.dumps(dict(path=self.folder.name)))))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                state=self.get_progress()
                self.assertTrue(state['active'])
                self.assertEqual((state['done'],state['total'],state['current'],state['reused']),(1,3,'sub/image.png',1))
                self.assertFalse(release.is_set())
            finally:
                release.set();worker.join(5)
        self.assertEqual(replies[0][0],200)
        self.assertFalse(self.get_progress()['active'])

    def test_failed_scan_clears_active_progress(self):
        with patch.object(self.server.session.library,'scan',side_effect=ValueError('test scan failure')):
            status,_=self.request('/api/library',json.dumps(dict(path=self.folder.name)))
        self.assertEqual(status,400)
        self.assertFalse(self.get_progress()['active'])

    def test_scan_progress_counts_failed_files_and_reused_cache(self):
        folder=Path(self.folder.name)/'progress-photos';folder.mkdir()
        Image.new('RGB',(30,30)).save(folder/'a.png')
        Image.new('RGB',(30,30)).save(folder/'b.png')
        (folder/'c.png').write_bytes(b'invalid image')
        library=Library(Path(self.folder.name)/'progress-cache')
        events=[]
        first=library.scan(folder,Encoder(),progress=lambda **state:events.append(state))
        self.assertEqual((first['files'],len(first['errors'])),(2,1))
        self.assertEqual((events[-1]['done'],events[-1]['total'],events[-1]['errors']),(3,3,1))
        self.assertIn('c.png',[state.get('current') for state in events])
        events.clear()
        library.scan(folder,Encoder(),progress=lambda **state:events.append(state))
        self.assertEqual(events[-1]['reused'],2)

    def test_speech_scan_and_prefix_bypass_clip_and_preserve_music_words_as_literal(self):
        folder=Path(self.folder.name)/'audio';folder.mkdir();(folder/'a.wav').write_bytes(b'fixture')
        self.server.session.encoder=None
        with patch('app.ClipEncoder',side_effect=AssertionError('Do not load CLIP')):
            status,result=self.request('/api/library',json.dumps(dict(path=str(folder),target='speech')))
        self.assertEqual(status,200);self.assertEqual(result['audio'],1)
        engine=unittest.mock.Mock();engine.search.return_value=dict(files=[],total_files=0,search_method='speech')
        self.server.session.speech_search=engine
        for target,query in [('speech','这首歌很好听'),('auto','台词：这首歌很好听'),('auto','台词:这首歌很好听')]:
            status,result=self.request('/api/library/search',json.dumps(dict(query=query,target=target)))
            self.assertEqual(status,200);self.assertEqual(engine.search.call_args.args[1],'这首歌很好听')
        status,_=self.request('/api/library/search',json.dumps(dict(query='开会',target='speech',threshold=.2)))
        self.assertEqual(status,400)
        status,_=self.request('/api/search',json.dumps(dict(query='开会',target='speech')))
        self.assertEqual(status,400)

    def test_upload_search_thumbnail_and_empty_results(self):
        status, result = self.request('/api/load', self.video)
        self.assertEqual(status, 200)
        self.assertEqual(result['duration'], 2)
        status, result = self.request('/api/search', json.dumps(dict(query='white screen', threshold=.5)))
        self.assertEqual(status, 200)
        scene = result['scenes'][0]
        self.assertEqual((scene['start'], scene['end']), (1, 2))
        self.assertEqual(scene['peak_time'],1)
        self.assertTrue(scene['thumbnail'].startswith('data:image/jpeg;base64,'))
        _, result = self.request('/api/search', json.dumps(dict(query='dog', threshold=.5)))
        self.assertEqual(result['scenes'], [])

    def test_failed_upload_preserves_existing_index(self):
        self.request('/api/load', self.video)
        previous = self.server.session.path
        status, _ = self.request('/api/load', b'not a video')
        self.assertEqual(status, 400)
        self.assertEqual(self.server.session.path, previous)
        self.assertTrue(previous.exists())

    def test_cross_origin_request_is_rejected(self):
        status, _ = self.request('/api/load', self.video, 'https://example.com')
        self.assertEqual(status, 403)
        self.assertIsNone(self.server.session.index)

    def test_object_search_uses_detector_and_its_own_default_threshold(self):
        session=self.server.session
        session.library.items={'photo':dict(path=Path(self.folder.name)/'photo.png',kind='image')}
        detector=unittest.mock.Mock()
        detector.search.return_value=dict(files=[],total_files=0,scenes=[],total=0)
        session.object_search=detector
        status,_=self.request('/api/library/search',json.dumps(dict(query='人',target='object')))
        self.assertEqual(status,200)
        self.assertEqual(detector.search.call_args.args[1:3],('人',OBJECT_THRESHOLD))
        status,_=self.request('/api/search',json.dumps(dict(query='人',target='object')))
        self.assertEqual(status,400)

    def test_page_uses_the_active_object_profile_threshold(self):
        with patch('app.OBJECT_THRESHOLD',.8):
            connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
            connection.request('GET','/')
            response=connection.getresponse()
            self.assertEqual(response.status,200)
            self.assertIn('data-object-threshold="0.80"',response.read().decode('utf-8'))
            connection.close()

    def test_auto_routes_photo_category_to_detection_with_its_own_threshold(self):
        session=self.server.session
        session.library.items={'photo':dict(path=Path(self.folder.name)/'photo.png',kind='image')}
        detector=unittest.mock.Mock()
        detector.search.return_value=dict(files=[],total_files=0,scenes=[],total=0)
        session.object_search=detector
        status,result=self.request('/api/library/search',json.dumps(dict(query='猫',target='auto')))
        self.assertEqual(status,200)
        self.assertEqual(detector.search.call_args.args[1:3],('猫',OBJECT_THRESHOLD))
        self.assertEqual(result['search_method'],'object')
        self.assertEqual(result['score_label'],'检测置信度')
        self.assertIn('猫',result['match_reason'])
        self.assertIn('自动选择物体检测',result['notice'])

    def test_auto_keeps_compound_queries_and_mixed_libraries_on_visual_search(self):
        session=self.server.session
        session.library.items={'photo':dict(kind='image')}
        with patch.object(session.library,'search',return_value=dict(files=[],total_files=0,scenes=[])) as search:
            status,result=self.request('/api/library/search',json.dumps(dict(query='雪地里奔跑的狗',target='auto')))
            self.assertEqual(status,200)
            self.assertEqual(result['search_method'],'visual')
            self.assertEqual(search.call_args.args[1],.2)
            self.assertIsNone(session.object_search)
            session.library.items['video']=dict(kind='video')
            status,result=self.request('/api/library/search',json.dumps(dict(query='猫',target='auto')))
            self.assertEqual(status,200)
            self.assertEqual(result['search_method'],'visual')
            self.assertIn('含视频',result['notice'])
            self.assertIsNone(session.object_search)

    def test_auto_condition_route_preserves_review_group_and_score_meaning(self):
        session=self.server.session
        session.library.items={'photo':dict(kind='image')}
        session.object_search=unittest.mock.Mock()
        response=dict(files=[],total_files=0,scenes=[],total=0,threshold=.2,
            review_files=[dict(name='possible.jpg')],review_total_files=1,
            score_label='整句画面相似度',match_reason='条件候选：雪地 ＋ 狗')
        with patch('app.search_conditions',return_value=response) as search:
            status,result=self.request('/api/library/search',json.dumps(dict(query='雪地里的狗',target='auto')))
        self.assertEqual(status,200)
        self.assertEqual(result['search_method'],'conditions')
        self.assertEqual(result['score_label'],'整句画面相似度')
        self.assertEqual(result['review_total_files'],1)
        self.assertEqual(search.call_args.args[2],dict(context='雪地',object='狗'))

    def test_auto_single_video_stays_visual_and_rejects_ambiguous_threshold_override(self):
        self.request('/api/load',self.video)
        status,result=self.request('/api/search',json.dumps(dict(query='猫',target='auto')))
        self.assertEqual(status,200)
        self.assertEqual(result['search_method'],'visual')
        self.assertIn('单视频',result['notice'])
        status,_=self.request('/api/search',json.dumps(dict(query='猫',target='auto',threshold=.2)))
        self.assertEqual(status,400)
        status,_=self.request('/api/search',json.dumps(dict(query='这首歌',target='auto')))
        self.assertEqual(status,400)

    def test_mixed_folder_and_video_range(self):
        Image.new('RGB', (32, 32), 'white').save(Path(self.folder.name) / 'photo.png')
        status, result = self.request('/api/library', json.dumps(dict(path=self.folder.name)))
        self.assertEqual(status, 200)
        self.assertEqual((result['photos'], result['videos']), (1, 1))
        status, result = self.request('/api/library/search', json.dumps(dict(query='white screen', threshold=.5)))
        self.assertEqual(status, 200)
        self.assertEqual({s['kind'] for s in result['scenes']}, {'image', 'video'})
        self.assertEqual(result['total_files'],2)
        video = next(s for s in result['scenes'] if s['kind'] == 'video')
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        connection.request('GET', '/media/' + video['file_id'], headers={'Range': 'bytes=0-9'})
        response = connection.getresponse()
        self.assertEqual(response.status, 206)
        self.assertEqual(response.read(), self.video[:10])
        connection.close()

    def test_grouping_precedes_scene_limit(self):
        matches=[dict(file_id='a',name='a.mp4',kind='video',start=i,end=i+1,score=.9) for i in range(60)]
        matches.append(dict(file_id='b',name='b.mp4',kind='video',start=0,end=1,score=.8))
        items={'a':dict(path=Path('a.mp4')),'b':dict(path=Path('b.mp4'))}
        with patch('library.thumbnail',return_value=None):
            result=file_results(matches,items)
        self.assertEqual(result['total_files'],2)
        self.assertEqual([f['name'] for f in result['files']],['a.mp4','b.mp4'])
        self.assertEqual(result['files'][0]['segment_count'],60)
        self.assertEqual(len(result['files'][0]['segments']),50)

    def test_file_card_preserves_peak_for_preview_and_segments(self):
        matches=[dict(file_id='a',name='a.mp4',kind='video',start=1,end=5,score=.9,peak_time=3)]
        with patch('library.thumbnail',return_value=None) as thumbnail:
            result=file_results(matches,{'a':dict(path=Path('a.mp4'))})
        thumbnail.assert_called_once_with(Path('a.mp4'),'video',3)
        self.assertEqual(result['files'][0]['peak_time'],3)
        self.assertEqual(result['files'][0]['segments'][0]['peak_time'],3)

    def test_different_encoder_does_not_reuse_incompatible_vectors(self):
        library=self.server.session.library
        encoder=Encoder()
        self.assertEqual(library.scan(self.folder.name,encoder)['reused'],0)
        self.assertEqual(library.scan(self.folder.name,encoder)['reused'],1)
        encoder.cache_id='test-brightness-v2'
        self.assertEqual(library.scan(self.folder.name,encoder)['reused'],0)

    def test_reveal_uses_only_registered_file_id(self):
        self.request('/api/library',json.dumps(dict(path=self.folder.name)))
        key=next(iter(self.server.session.library.items))
        path=self.server.session.library.items[key]['path']
        with patch('app.subprocess.Popen') as launch:
            status,result=self.request('/api/reveal',json.dumps(dict(file_id=key)))
            self.assertEqual(status,200)
            launch.assert_called_once_with(['explorer.exe','/select,',str(path)])
        with patch('app.subprocess.Popen') as launch:
            status,_=self.request('/api/reveal',json.dumps(dict(file_id='../../private')))
            self.assertEqual(status,400)
            launch.assert_not_called()

    def test_cache_survives_new_library_and_tracks_changes(self):
        root = Path(self.folder.name)
        nested = root / 'nested'
        nested.mkdir()
        for folder in (root, nested):
            Image.new('RGB', (32, 32), 'white').save(folder / 'same.png')
        (root / 'broken.jpg').write_bytes(b'not an image')
        library = self.server.session.library
        first = library.scan(root, Encoder())
        self.assertEqual((first['files'], first['reused'], len(first['errors'])), (3, 0, 1))
        library = Library(library.cache)
        second = library.scan(root, Encoder())
        self.assertEqual(second['reused'], 3)
        matches = library.search('white screen', .5, Encoder())['scenes']
        names = {s['name'] for s in matches}
        self.assertIn('same.png', names)
        self.assertIn(str(Path('nested') / 'same.png'), names)
        (nested / 'same.png').unlink()
        Image.new('RGB', (32, 32), 'black').save(root / 'same.png')
        changed = library.scan(root, Encoder())
        self.assertEqual((changed['files'], changed['reused']), (2, 1))
        self.assertEqual([s['name'] for s in library.search('white screen', .5, Encoder())['scenes']], ['input.avi'])


if __name__ == '__main__':
    unittest.main()
