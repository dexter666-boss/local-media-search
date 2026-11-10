"""Local browser UI for the learning project. Run with the project venv."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import subprocess
import os
from pathlib import Path
import tempfile
import threading

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['TORCH_HOME'] = str(ROOT / '.cache' / 'torch')
from search import ClipEncoder, index_video, refine_scenes
from library import Library
from text_search import TextSearch
from speech_search import SpeechSearch
from object_search import ObjectSearch
from condition_search import parse_conditions,search_conditions,CONTEXT_THRESHOLD

OBJECT_QUALITY=os.environ.get('MEDIA_SEARCH_OBJECT_QUALITY','fast')
OBJECT_THRESHOLD=ObjectSearch.threshold_for(OBJECT_QUALITY)


class Session:
    def __init__(self):
        self.encoder = None
        self.object_search = None
        self.text_search = TextSearch()
        self.speech_search = SpeechSearch()
        self.index = None
        self.path = None
        self.lock = threading.Lock()
        self.library = Library(ROOT / '.cache' / 'library')
        self.progress = dict(active=False)

    def report_progress(self, phase, done=0, total=None, current='', reused=0, errors=0):
        # Replace the snapshot atomically; progress reads must not wait on the task lock.
        self.progress = dict(active=True,phase=phase,done=done,total=total,
                             current=current,reused=reused,errors=errors)

    def load(self, path):
        if self.encoder is None:
            self.encoder = ClipEncoder()
        index = index_video(path, .5, self.encoder)
        self.path, self.index = path, index
        return dict(duration=index[2], frames=len(index[0]))

    def search(self, query, threshold):
        from query_support import validate_visual_query
        validate_visual_query(query)
        import cv2
        if self.index is None:
            raise ValueError('请先选择视频并等待索引完成。')
        times, vectors, duration, step = self.index
        scores = self.encoder.scores(vectors, query)
        scenes = refine_scenes(times, scores, duration, step, threshold, 1, 0)
        cap = cv2.VideoCapture(str(self.path))
        try:
            for scene in scenes:
                cap.set(cv2.CAP_PROP_POS_MSEC, scene['peak_time'] * 1000)
                ok, frame = cap.read()
                scene['thumbnail'] = None
                if ok:
                    height, width = frame.shape[:2]
                    frame = cv2.resize(frame, (max(1, round(width * min(320 / width, 180 / height))),
                                               max(1, round(height * min(320 / width, 180 / height)))))
                    ok, jpeg = cv2.imencode('.jpg', frame)
                    if ok:
                        scene['thumbnail'] = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
        finally:
            cap.release()
        return dict(scenes=scenes, duration=duration, threshold=threshold)


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, kind='application/json; charset=utf-8'):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            page=(ROOT/'ui.html').read_text(encoding='utf-8').replace(
                'data-object-threshold="0.50"',f'data-object-threshold="{OBJECT_THRESHOLD:.2f}"')
            self.reply(200,page.encode('utf-8'),'text/html; charset=utf-8')
        elif self.path == '/api/progress':
            self.reply(200,self.server.session.progress)
        elif self.path.startswith('/media/'):
            item = self.server.session.library.items.get(self.path.removeprefix('/media/'))
            if not item or not item['path'].is_file():
                self.reply(404, dict(error='素材已不存在，请重新扫描文件夹。'))
                return
            path = item['path']
            size = path.stat().st_size
            start, end = 0, size - 1
            partial = self.headers.get('Range')
            if partial:
                try:
                    unit, positions = partial.split('=', 1)
                    first, last = positions.split('-', 1)
                    if unit != 'bytes' or ',' in positions:
                        raise ValueError()
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                    if not 0 <= start <= end < size:
                        raise ValueError()
                except ValueError:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with path.open('rb') as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.reply(404, dict(error='Not found'))

    def do_POST(self):
        expected_origin = f'http://127.0.0.1:{self.server.server_port}'
        if self.headers.get('Origin') not in (None, expected_origin):
            self.reply(403, dict(error='只接受本机界面请求。'))
            return
        session = self.server.session
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0:
                raise ValueError('请求内容为空。')
            with session.lock:
                session.report_progress('准备处理')
                try:
                    if self.path == '/api/load':
                        # Keep the previous video usable if decoding the new file fails.
                        with tempfile.NamedTemporaryFile(dir=self.server.uploads, suffix='.video', delete=False) as output:
                            path = Path(output.name)
                            remaining = length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError('视频传输未完成，请重新选择。')
                                output.write(chunk)
                                remaining -= len(chunk)
                        previous = session.path
                        try:
                            result = session.load(path)
                        except Exception:
                            path.unlink(missing_ok=True)
                            raise
                        if previous:
                            previous.unlink(missing_ok=True)
                    elif self.path == '/api/folder':
                        self.rfile.read(length)
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        try:
                            result = dict(path=filedialog.askdirectory(parent=root, title='选择照片和视频素材文件夹'))
                        finally:
                            root.destroy()
                    elif self.path == '/api/reveal':
                        if length > 16384:
                            raise ValueError('请求过长。')
                        body = json.loads(self.rfile.read(length))
                        item = session.library.items.get(body.get('file_id'))
                        if not item or not item['path'].is_file():
                            raise ValueError('素材不存在，请重新扫描。')
                        subprocess.Popen(['explorer.exe', '/select,', str(item['path'])])
                        result = dict(ok=True)
                    elif self.path == '/api/library':
                        if length > 16384:
                            raise ValueError('文件夹路径过长。')
                        body = json.loads(self.rfile.read(length))
                        if not body.get('path', '').strip():
                            raise ValueError('请选择或输入素材文件夹。')
                        text_only=body.get('target') in ('text','speech')
                        if not text_only and session.encoder is None:
                            session.report_progress('加载画面模型，首次使用可能需要下载')
                            session.encoder = ClipEncoder()
                        result = session.library.scan(body['path'], None if text_only else session.encoder,progress=session.report_progress,**({'speech_only':True} if body.get('target')=='speech' else {}))
                        result['indexing']=body.get('target') if text_only else 'visual'
                    elif self.path in ('/api/search', '/api/library/search'):
                        if length > 16384:
                            raise ValueError('搜索描述过长。')
                        body = json.loads(self.rfile.read(length))
                        query = body.get('query', '').strip()
                        from query_support import validate_visual_query
                        target=body.get('target','visual')
                        if target=='auto' and query.startswith(('文字：','文字:')):
                            target='text'
                            query=query[3:].strip()
                        if target=='auto' and query.startswith(('台词：','台词:')):
                            target='speech';query=query[3:].strip()
                        if target=='speech':
                            if self.path!='/api/library/search' or not session.library.items:
                                raise ValueError('语音搜索请先切换文件夹模式并扫描音视频。')
                            if 'threshold' in body:raise ValueError('语音原文搜索不使用相似度阈值。')
                            result=session.speech_search.search(session.library.items,query,ROOT/'.cache/speech',progress=session.report_progress)
                            self.reply(200,result)
                            return
                        if target=='text':
                            if self.path!='/api/library/search' or not session.library.items:
                                raise ValueError('文字搜索请先切换文件夹模式并扫描图片。')
                            if 'threshold' in body:
                                raise ValueError('文字搜索按原文包含匹配，不使用相似度阈值。')
                            result=session.text_search.search(session.library.items,query,ROOT/'.cache/ocr',progress=session.report_progress)
                            self.reply(200,result)
                            return
                        automatic=target=='auto'
                        route_notice=''
                        condition_plan=None
                        if any('index' in item and item['index'] is None for item in session.library.items.values()) and self.path=='/api/library/search':
                            raise ValueError('当前仅登记了文字素材，请切换检索方式后重新扫描以建立画面索引。')
                        if automatic:
                            if 'threshold' in body:
                                raise ValueError('自动检索使用对应能力的推荐阈值；如需调整，请选择画面检索或物体检索。')
                            photos_only=(self.path=='/api/library/search' and bool(session.library.items)
                                         and all(item['kind']=='image' for item in session.library.items.values()))
                            condition_plan=parse_conditions(query) if photos_only else None
                            target='conditions' if condition_plan else 'object' if photos_only and ObjectSearch.supports(query) else 'visual'
                            if target=='conditions':
                                route_notice='已分别筛选画面条件与物体，未全部通过的相似素材另列待核对；'
                            elif target=='object':
                                route_notice='已自动选择物体检测；'
                            elif self.path=='/api/search':
                                route_notice='单视频按画面相似度定位，尚未进行逐帧物体核对；'
                            elif any(item['kind']=='video' for item in session.library.items.values()):
                                route_notice='当前文件夹含视频，按画面相似度检索，尚未逐项核对物体；'
                            else:
                                route_notice='已自动选择画面相似检索，复杂描述仍需预览核对；'
                        validate_visual_query(query,'visual' if target=='object' or (automatic and target=='conditions') else target)
                        threshold = float(body.get('threshold', OBJECT_THRESHOLD if target=='object' else .20))
                        if not query:
                            raise ValueError('请输入英文描述。')
                        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
                            raise ValueError('相似度阈值应在 -1 到 1 之间。')
                        if target in ('object','conditions'):
                            if self.path!='/api/library/search' or not session.library.items:
                                raise ValueError('请先扫描纯图片文件夹，再使用物体检索。')
                            if not 0 <= threshold <= 1:
                                raise ValueError('检测置信度应在 0 到 1 之间。')
                            if session.object_search is None:
                                session.object_search=ObjectSearch(quality=OBJECT_QUALITY)
                            if target=='conditions':
                                result=search_conditions(session.library.items,query,condition_plan,session.encoder,
                                    session.object_search,ROOT/'.cache/objects',CONTEXT_THRESHOLD)
                            else:
                                result=session.object_search.search(session.library.items,query,threshold,ROOT/'.cache/objects')
                        elif self.path == '/api/library/search':
                            if not session.library.items:
                                raise ValueError('请先扫描包含照片或视频的文件夹。')
                            result = session.library.search(query, threshold, session.encoder)
                        else:
                            result = session.search(query, threshold)
                        result['search_method']=target
                        if target!='conditions':
                            result['score_label']='检测置信度' if target=='object' else '画面相似度'
                            result['match_reason']='物体检测候选：'+query if target=='object' else '与描述画面相似，未逐项确认全部条件'
                        if automatic:
                            result['notice']=route_notice+result.get('notice','搜索完成，请预览核对候选。')
                    else:
                        self.reply(404, dict(error='Not found'))
                        return
                finally:
                    session.progress = dict(session.progress,active=False)
            self.reply(200, result)
        except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
            self.reply(400, dict(error=str(error)))


def main(open_browser=False):
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ui-', dir=cache) as uploads:
        server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
        server.session = Session()
        server.uploads = uploads
        print('打开 http://127.0.0.1:8765 ，按 Ctrl+C 停止。', flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open('http://127.0.0.1:8765/')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
