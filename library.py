"""Folder indexing with local, non-pickle embedding caches."""
import base64
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps
from search import index_video, refine_scenes

IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
AUDIO = {'.wav','.mp3','.m4a','.flac','.ogg','.aac','.wma','.opus'}
VIDEOS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


def thumbnail(path, kind, timestamp=0):
    if kind == 'image':
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
    else:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                return None
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    import io
    image.thumbnail((320, 180))
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()


def file_results(matches, items):
    """Group before display limits so a long video's hits cannot hide other files."""
    grouped = {}
    for scene in matches:
        key = scene['file_id']
        if key not in grouped:
            grouped[key] = dict(scene, path=str(items[key]['path']), segments=[], segment_count=0)
        file = grouped[key]
        file['segment_count'] += 1
        if len(file['segments']) < 50:
            file['segments'].append({k: scene[k] for k in ('start','end','score','peak_time','text') if k in scene})
    files = list(grouped.values())[:50]
    for file in files:
        if file['kind']!='audio' and not file.get('thumbnail'):
            try:
                file['thumbnail'] = thumbnail(items[file['file_id']]['path'],file['kind'],file.get('peak_time',file['start']) or 0)
            except (OSError, ValueError):
                file['thumbnail'] = None
        file['segments'].sort(key=lambda scene: scene['start'] or 0)
    return dict(files=files,total_files=len(grouped))


class Library:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.items = {}

    def scan(self, folder, encoder, progress=None, speech_only=False):
        root = Path(folder).resolve()
        if not root.is_dir():
            raise ValueError('文件夹不存在。')
        self.cache.mkdir(parents=True, exist_ok=True)
        items, errors, reused = {}, [], 0
        if progress:
            progress(phase='枚举素材文件')
        paths=[path for path in sorted(root.rglob('*'))
               if path.is_file() and path.suffix.lower() in (VIDEOS | AUDIO if speech_only else IMAGES | VIDEOS)]
        for number,path in enumerate(paths):
            if progress:
                progress(phase='登记文件' if encoder is None else '建立画面索引',done=number,
                         total=len(paths),current=str(path.relative_to(root)),reused=reused,errors=len(errors))
            try:
                kind = 'audio' if path.suffix.lower() in AUDIO else 'image' if path.suffix.lower() in IMAGES else 'video'
                if encoder is None:
                    file_id = hashlib.sha256(str(path).encode()).hexdigest()
                    items[file_id] = dict(path=path, name=str(path.relative_to(root)), kind=kind, index=None)
                    continue
                stat = path.stat()
                # Changing preprocessing/model requires a new cache version.
                signature = f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{encoder.cache_id}|rgb-exif-grid05-v2'
                cache_file = self.cache / (hashlib.sha256(signature.encode()).hexdigest() + '.npz')
                kind = 'image' if path.suffix.lower() in IMAGES else 'video'
                index = None
                if cache_file.exists():
                    try:
                        with np.load(cache_file, allow_pickle=False) as saved:
                            index = (saved['times'].tolist(), encoder.torch.from_numpy(saved['vectors'].copy()),
                                     float(saved['duration']), float(saved['step']))
                        reused += 1
                    except (ValueError, OSError, KeyError):
                        index = None
                if index is None:
                    if kind == 'image':
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert('RGB')
                            vectors = encoder.images([image])
                        index = ([0.], vectors, 0., .5)
                    else:
                        index = index_video(path, .5, encoder)
                    vectors = index[1]
                    array = vectors.numpy() if hasattr(vectors, 'numpy') else np.asarray(vectors)
                    with tempfile.NamedTemporaryFile(dir=self.cache, suffix='.npz', delete=False) as temp:
                        temporary = Path(temp.name)
                        np.savez(temp, times=index[0], vectors=array, duration=index[2], step=index[3])
                    os.replace(temporary, cache_file)
                file_id = hashlib.sha256(str(path).encode()).hexdigest()
                items[file_id] = dict(path=path, name=str(path.relative_to(root)), kind=kind, index=index)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(dict(name=str(path.relative_to(root)), error=str(error)))
        if progress:
            progress(phase='扫描完成',done=len(paths),total=len(paths),reused=reused,errors=len(errors))
        self.items = items
        return dict(files=len(items), photos=sum(item['kind'] == 'image' for item in items.values()),
                    videos=sum(item['kind'] == 'video' for item in items.values()),
                    audio=sum(item['kind']=='audio' for item in items.values()), reused=reused, errors=errors)

    def search(self, query, threshold, encoder):
        from query_support import validate_visual_query
        validate_visual_query(query)
        matches = []
        for file_id, item in self.items.items():
            if item['index'] is None:
                raise ValueError('当前为文字文件列表，请切换检索方式后重新扫描以建立画面索引。')
            times, vectors, duration, step = item['index']
            scores = encoder.scores(vectors, query)
            if item['kind'] == 'image':
                scenes = [dict(start=None, end=None, score=scores[0])] if scores[0] >= threshold else []
            else:
                scenes = refine_scenes(times, scores, duration, step, threshold, 1, 0)
            for scene in scenes:
                matches.append(dict(scene, file_id=file_id, name=item['name'], kind=item['kind']))
        matches.sort(key=lambda scene: scene['score'], reverse=True)
        total = len(matches)
        for scene in matches[:50]:
            item = self.items[scene['file_id']]
            try:
                scene['thumbnail'] = thumbnail(item['path'], item['kind'], scene.get('peak_time',scene['start']) or 0)
            except (OSError, ValueError):
                scene['thumbnail'] = None
        return dict(scenes=matches[:50], total=total, threshold=threshold, **file_results(matches,self.items))
