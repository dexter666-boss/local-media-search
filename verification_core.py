"""Conservative handling of model opinions, independent of topic names."""
import json
import re


def json_object(text):
    try:
        value, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
        return value if isinstance(value,dict) else {}
    except (ValueError, json.JSONDecodeError):
        return {}


def normalized(text):
    return re.sub(r'[^\w\u3400-\u9fff]','',text.casefold())


def grounded_decision(raw, plan, observation):
    """A query-conditioned answer cannot invent a named identity as evidence."""
    evidence=[]
    for field in ('objects','events'):
        values=observation.get(field,[])
        if isinstance(values,list):
            evidence.extend(item for item in values if isinstance(item,str))
    names=plan.get('named_entities')
    if not isinstance(names,list) or not all(isinstance(item,str) for item in names):
        return dict(verdict='uncertain',evidence=evidence,missing=['无法可靠解析查询要求。'])
    readable=observation.get('readable_text',[])
    entities=observation.get('named_entities',[])
    if not isinstance(readable,list) or not isinstance(entities,list):
        readable,entities=[],[]
    observed={normalized(item) for item in readable+entities if isinstance(item,str)}
    missing=[name for name in names if normalized(name) not in observed]
    if missing:
        return dict(verdict='uncertain',evidence=evidence,missing=['未获得独立画面证据：'+name for name in missing])
    verdict=raw.get('verdict')
    reasons=raw.get('evidence',[])
    absent=raw.get('missing',[])
    if (verdict not in ('match','reject','uncertain') or not isinstance(reasons,list)
            or not isinstance(absent,list) or not all(isinstance(v,str) for v in reasons+absent)):
        return dict(verdict='uncertain',evidence=evidence,missing=['核对模型没有返回有效判断。'])
    if verdict=='match' and (absent or not reasons):
        verdict='uncertain'
    return dict(verdict=verdict,evidence=reasons or evidence,missing=absent)
