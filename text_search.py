"""Local OCR with cached, literal text matching and original-image coordinates."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

import numpy as np
from PIL import Image, ImageOps
from library import file_results


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def matching_lines(lines, query):
    """Ignore whitespace, including wrapped lines; preserve punctuation and word order."""
    needle=normalize(query)
    if not needle:
        raise ValueError('请输入要查找的文字。')
    parts=[normalize(line['text']) for line in lines]
    text=''.join(parts)
    spans=[]
    start=text.find(needle)
    while start>=0:
        spans.append((start,start+len(needle)))
        start=text.find(needle,start+1)
    result=[]
    offset=0
    for line,part in zip(lines,parts):
        end=offset+len(part)
        if any(left<end and right>offset for left,right in spans):
            result.append(line)
        offset=end
    return result


class TextSearch:
    cache_id='rapidocr-onnxruntime-1.4.4:ppocr-default:exif-rgb:1600-max:4096:v1'

    def __init__(self):
        self.engine=None

    def recognize(self, path):
        if self.engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self.engine=RapidOCR(det_limit_type='max',det_limit_side_len=1600,
                                 max_side_len=4096,intra_op_num_threads=4,inter_op_num_threads=1)
        with Image.open(path) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
            width,height=image.size
            # RapidOCR ndarray inputs use BGR; convert after EXIF orientation correction.
            result,_=self.engine(np.asarray(image)[:,:,::-1].copy())
        lines=[dict(text=text,confidence=float(score),
                    box=[[float(x)/width,float(y)/height] for x,y in box])
               for box,text,score in (result or [])]
        return dict(size=[width,height],lines=lines)

    def document(self, path, cache):
        path=Path(path)
        stat=path.stat()
        signature=f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.cache_id}'
        cache=Path(cache)
        cache.mkdir(parents=True,exist_ok=True)
        target=cache/(hashlib.sha256(signature.encode()).hexdigest()+'.json')
        if target.exists():
            try:
                document=json.loads(target.read_text(encoding='utf-8'))
                if isinstance(document['lines'],list) and len(document['size'])==2:
                    return document,True
            except (OSError,ValueError,KeyError,TypeError):
                pass
        document=self.recognize(path)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=cache,
                                         suffix='.json',delete=False) as output:
            temporary=Path(output.name)
            json.dump(document,output,ensure_ascii=False)
        os.replace(temporary,target)
        return document,False

    def search(self, items, query, cache, progress=None):
        if not normalize(query):
            raise ValueError('请输入要查找的文字。')
        matches,errors=[],[]
        reused=processed=skipped=0
        photos=[(file_id,item) for file_id,item in items.items() if item['kind']=='image']
        skipped=len(items)-len(photos)
        for number,(file_id,item) in enumerate(photos):
            if progress:
                progress(phase='识别图片文字',done=number,total=len(photos),current=item['name'],
                         reused=reused,errors=len(errors))
            try:
                document,hit=self.document(item['path'],cache)
                reused+=int(hit)
                processed+=1
                lines=matching_lines(document['lines'],query)
                if lines:
                    matches.append(dict(file_id=file_id,name=item['name'],kind='image',
                                        start=None,end=None,score=len(lines),ocr_lines=lines,
                                        ocr_size=document['size']))
            except (OSError,ValueError,RuntimeError) as error:
                errors.append(dict(name=item['name'],error=str(error)))
        if progress:
            progress(phase='整理文字结果',done=len(photos),total=len(photos),reused=reused,errors=len(errors))
        matches.sort(key=lambda file:(-file['score'],file['name']))
        result=file_results(matches,items)
        result.update(search_method='text',score_label='匹配文字行数',match_reason='OCR 原文包含查询文字',
                      ocr_processed=processed,ocr_reused=reused,skipped_videos=skipped,errors=errors,
                      notice=f'已识别或读取 {processed} 张图片（{reused} 张复用缓存），跳过 {skipped} 个视频，'
                             f'{len(errors)} 张处理失败。按识别原文包含匹配，忽略空白和英文大小写；'
                             '保留标点，不匹配近义句。框选命中的整行，请核对截图。')
        return result
