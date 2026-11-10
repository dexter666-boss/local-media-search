"""Reject unsupported search types rather than return visual-score guesses."""
import re


def validate_visual_query(query, target='visual'):
    if not query.strip():
        raise ValueError('请输入搜索描述。')
    audio_intent = re.search(
        r'背景音乐|歌曲|这首歌|歌名|音轨|配乐|\b(background music|song named|song called|soundtrack|lyrics|bgm)\b',
        query, re.IGNORECASE)
    if target != 'visual' or audio_intent:
        raise ValueError('当前画面检索不识别歌曲；讲话内容请使用语音台词检索，不能按歌名或背景音乐筛选。')
