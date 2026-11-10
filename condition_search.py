"""Separate scene similarity and object presence for simple positive photo queries."""
import re

from library import file_results
from object_search import ObjectSearch

CONTEXT_THRESHOLD=.08


def parse_conditions(query):
    match=re.fullmatch(r'(?:在)?([\u3400-\u9fff]{2,16}?)(?:里面|上面|里|上|内)的([\u3400-\u9fff]{1,8})',query.strip())
    if not match:
        return None
    context,category=match.groups()
    # Keep negation, conjunctions and relative-object positions on the existing path.
    if re.search(r'的|不|没|无|非|和|与|或|旁|左|右|前|后|正在',context):
        return None
    if not ObjectSearch.supports(category) or ObjectSearch.supports(context):
        return None
    return dict(context=context,object=category)


def search_conditions(items,query,plan,encoder,detector,cache,context_threshold,context_template='{context}'):
    if any(item['kind']!='image' for item in items.values()):
        raise ValueError('条件筛选目前仅支持纯图片文件夹。')
    if not items:
        raise ValueError('请先扫描图片文件夹。')
    keys=list(items)
    vectors=encoder.torch.cat([items[key]['index'][1] for key in keys])
    scores=dict(zip(keys,encoder.scores(vectors,context_template.format(context=plan['context']))))
    query_scores=dict(zip(keys,encoder.scores(vectors,query)))
    selected={key:items[key] for key in keys if query_scores[key]>=.2}
    # Filter every eligible photo before applying the UI's 50-card display limit.
    detections=detector.evidence(selected,plan['object'],detector.default_threshold,cache) if selected else []
    matches=[]
    review=[]
    for detection in detections:
        key=detection['file_id']
        conditions=[dict(kind='object',term=plan['object'],score=detection['score'],
                         threshold=detector.default_threshold,label='物体检测置信度',
                         passed=detection['score']>0 and detection['score']>=detector.default_threshold),
                    dict(kind='scene',term=plan['context'],score=scores[key],
                         threshold=context_threshold,label='场景画面相似度',passed=scores[key]>=context_threshold)]
        passed=all(row['passed'] for row in conditions)
        row=dict(detection,score=query_scores[key],conditions=conditions,
                 match_reason=f"条件候选：{plan['context']} ＋ {plan['object']}" if passed else '待核对：条件未全部通过')
        (matches if passed else review).append(row)
    matches.sort(key=lambda row:row['score'],reverse=True)
    review.sort(key=lambda row:row['score'],reverse=True)
    deferred=file_results(review,items)
    return dict(scenes=matches[:50],total=len(matches),threshold=.2,
        review_files=deferred['files'],review_total_files=deferred['total_files'],
        score_label='整句画面相似度',match_reason=f"条件候选：{plan['context']} ＋ {plan['object']}",
        notice='物体检测、场景相似度与整句描述已分别筛选；仍可能误检或漏检，未证明精确位置关系或专有身份，请预览核对。',
        **file_results(matches,items))
