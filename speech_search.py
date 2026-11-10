"""Cached local speech transcription and literal, timestamped retrieval."""
from functools import lru_cache
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import unicodedata
from library import file_results


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    converter=OpenCC('t2s')
    converter.convert('初始化')
    return converter


def normalize_simplified(text):
    return normalize(_converter().convert(unicodedata.normalize('NFKC',text)))


_NUMERALS='0-9零〇一二两三四五六七八九十百千万亿'
_QUANTITY=re.compile(r'(?<!['+_NUMERALS+r'.点第数几之负])'
                     r'([0-9零〇一二两三四五六七八九十]['+_NUMERALS+r']*)'
                     r'(公里|千米|厘米|毫米|公斤|千克|毫升|分钟|小时|元|年|月|日|天|分|秒|米|岁|个|次|人|点(?!['+_NUMERALS+r'也都不儿点]))')


def normalize_quantities(text):
    text=_converter().convert(unicodedata.normalize('NFKC',text))
    def replace(match):
        number,unit=match.groups()
        if not re.search('[零〇一二两三四五六七八九十百千万亿]',number):
            return match.group()
        # Keep identifiers with leading zeros and ambiguous colloquial forms literal.
        if (len(number)>1 and number[0] in '0零〇') or re.search('[百千万亿][一二三四五六七八九]$',number):
            return match.group()
        from cn2an import cn2an
        try:
            value=cn2an(number.replace('〇','零'),'smart')
        except ValueError:
            return match.group()
        return str(value)+unit
    return normalize(_QUANTITY.sub(replace,text))


def _find_matches(segments,query,normalizer):
    needle=normalizer(query)
    if not needle:
        raise ValueError('请输入台词中的文字或数字。')
    text='';ranges=[];previous=None
    for segment in segments:
        if previous is not None and segment['start']-previous>2:
            text+='\0'
        start=len(text);text+=normalizer(segment['text'])
        ranges.append((start,len(text),segment));previous=segment['end']
    results=[];seen=set();offset=text.find(needle)
    while offset>=0:
        hits=[s for left,right,s in ranges if left<offset+len(needle) and right>offset]
        key=(hits[0]['start'],hits[-1]['end'])
        if key not in seen:
            seen.add(key);results.append(dict(start=key[0],end=key[1],text=' '.join(s['text'] for s in hits),score=1))
        offset=text.find(needle,offset+1)
    return results


def find_matches(segments,query):
    # Preserve literal hits: phrase-aware conversion may treat a short query
    # differently from its surrounding text (for example, 乾 versus 乾隆).
    matches=_find_matches(segments,query,normalize)
    matches+=_find_matches(segments,query,normalize_simplified)
    if _QUANTITY.search(_converter().convert(unicodedata.normalize('NFKC',query))):
        matches+=_find_matches(segments,query,normalize_quantities)
    unique={(match['start'],match['end']):match for match in matches}
    return sorted(unique.values(),key=lambda match:(match['start'],match['end']))


class SpeechSearch:
    cache_id='faster-whisper:small:cpu-int8:vad:beam5:no-context:duration-v2'

    def __init__(self):
        self.model=None

    def transcribe(self,path):
        import av
        try:
            with av.open(str(path)) as container:
                if not container.streams.audio:
                    return dict(segments=[],status='no_audio')
        except av.error.FFmpegError as error:
            raise ValueError('音视频无法解码：'+str(error)) from error
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model=WhisperModel('small',device='cpu',compute_type='int8',cpu_threads=4,
                                   download_root=str(Path(__file__).resolve().parent/'.cache/speech-models'))
        try:
            segments,info=self.model.transcribe(str(path),beam_size=5,vad_filter=True,
                                                condition_on_previous_text=False)
            rows=[]
            for segment in segments:
                start=max(0.,float(segment.start));end=min(float(info.duration),float(segment.end))
                text=segment.text.strip()
                if text and end>start:
                    rows.append(dict(start=start,end=end,text=text))
        except av.error.FFmpegError as error:
            raise ValueError('音轨无法解码：'+str(error)) from error
        return dict(segments=rows,status='transcribed' if rows else 'no_speech',language=info.language)

    def document(self,path,cache):
        path=Path(path);stat=path.stat();cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.cache_id}'.encode()).hexdigest()
        target=cache/(key+'.json')
        if target.exists():
            try:
                doc=json.loads(target.read_text(encoding='utf-8'))
                if isinstance(doc['segments'],list) and doc['status'] in ('no_audio','no_speech','transcribed'):
                    return doc,True
            except (OSError,ValueError,KeyError,TypeError):pass
        doc=self.transcribe(path)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=cache,suffix='.json',delete=False) as stream:
            temporary=Path(stream.name);json.dump(doc,stream,ensure_ascii=False)
        os.replace(temporary,target)
        return doc,False

    def search(self,items,query,cache,progress=None):
        if not normalize(query):raise ValueError('请输入台词中的文字或数字。')
        media=[(key,item) for key,item in items.items() if item['kind'] in ('video','audio')]
        matches=[];errors=[];reused=0;no_audio=0;no_speech=0
        for number,(key,item) in enumerate(media):
            if progress:progress(phase='转写语音，首次加载可能下载模型',done=number,total=len(media),current=item['name'],reused=reused,errors=len(errors))
            try:
                doc,hit=self.document(item['path'],cache);reused+=int(hit)
                no_audio+=int(doc['status']=='no_audio');no_speech+=int(doc['status']=='no_speech')
                for match in find_matches(doc['segments'],query):
                    matches.append(dict(match,file_id=key,name=item['name'],kind=item['kind']))
            except (OSError,ValueError,RuntimeError) as error:
                errors.append(dict(name=item['name'],error=str(error)))
        result=file_results(matches,items)
        result.update(search_method='speech',score_label='原文匹配',match_reason='转写台词包含查询文字',
                      total_matches=len(matches),reused=reused,errors=errors,no_audio=no_audio,no_speech=no_speech,
                      notice=f'检查 {len(media)} 个音视频，{reused} 个复用缓存；{no_audio} 个无音轨，{no_speech} 个未转写出语音，{len(errors)} 个失败。'
                             '按转写文字匹配，自动统一简繁体和带单位的整数写法，忽略标点、空白和英文大小写；不匹配近义句，不识别歌名。时间为模型估计，请播放核对。')
        return result
