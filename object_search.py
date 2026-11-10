"""Explicit object-presence search for photo libraries, using standard COCO classes."""
import hashlib
import json
from pathlib import Path

# Translations of the model's 80 supported categories, not per-dataset rules.
CHINESE='人 自行车 汽车 摩托车 飞机 公交车 火车 卡车 船 红绿灯 消防栓 停车标志 停车计时器 长椅 鸟 猫 狗 马 羊 牛 大象 熊 斑马 长颈鹿 背包 雨伞 手提包 领带 行李箱 飞盘 滑雪板 单板滑雪板 球 风筝 棒球棒 棒球手套 滑板 冲浪板 网球拍 瓶子 酒杯 杯子 叉子 刀 勺子 碗 香蕉 苹果 三明治 橙子 西兰花 胡萝卜 热狗 披萨 甜甜圈 蛋糕 椅子 沙发 盆栽 床 餐桌 马桶 电视 笔记本电脑 鼠标 遥控器 键盘 手机 微波炉 烤箱 烤面包机 水槽 冰箱 书 时钟 花瓶 剪刀 泰迪熊 吹风机 牙刷'.split()


class ObjectSearch:
    model_id='torchvision:fasterrcnn_mobilenet_v3_large_320_fpn:COCO_V1:v1'
    accurate_model_id='torchvision:fasterrcnn_resnet50_fpn_v2:COCO_V1:v1'

    @staticmethod
    def supports(query):
        query=query.strip().casefold()
        if query in CHINESE:
            return True
        if not query.isascii():
            return False
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        return any(query==name.casefold() for name in FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta['categories']
                   if name not in ('__background__','N/A'))

    @staticmethod
    def threshold_for(quality):
        if quality not in ('fast','accurate'):
            raise ValueError('未知物体检测配置。')
        return .8 if quality=='accurate' else .5

    def __init__(self,quality='fast'):
        self.default_threshold=self.threshold_for(quality)
        self.verify_dogs=quality=='accurate'
        self.dog_verifier=None
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as Weights
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        import torch
        self.torch=torch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        if quality=='accurate':
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            factory=fasterrcnn_resnet50_fpn_v2
            self.model_id=self.accurate_model_id
        else:
            weights=Weights.DEFAULT
            factory=fasterrcnn_mobilenet_v3_large_320_fpn
        categories=[(index,name) for index,name in enumerate(weights.meta['categories'])
                    if index and name!='N/A']
        assert len(CHINESE)==len(categories)
        self.names={}
        for chinese,(index,english) in zip(CHINESE,categories):
            self.names[chinese]=self.names[english]=index
        self.transform=weights.transforms()
        self.model=factory(weights=weights,progress=False).to(self.device).eval()

    def evidence(self,items,query,threshold,cache):
        from PIL import Image,ImageOps
        category=self.names.get(query.strip().casefold())
        if category is None:
            raise ValueError('物体检索需要输入单个支持的类别，例如：人、猫、狗、汽车、椅子。复杂场景请使用画面内容检索。')
        if any(item['kind']!='image' for item in items.values()):
            raise ValueError('物体检索目前支持纯图片文件夹；包含视频时请使用画面内容检索。')
        cache=Path(cache)
        cache.mkdir(parents=True,exist_ok=True)
        matches=[]
        for key,item in items.items():
            path=item['path']
            stat=path.stat()
            signature=f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.model_id}'
            saved=cache/(hashlib.sha256(signature.encode()).hexdigest()+'.json')
            if saved.exists():
                scores=json.loads(saved.read_text(encoding='utf-8'))
            else:
                with Image.open(path) as source:
                    image=ImageOps.exif_transpose(source).convert('RGB')
                    tensor=self.transform(image).to(self.device)
                with self.torch.inference_mode():
                    result=self.model([tensor])[0]
                scores={}
                for label,score in zip(result['labels'].tolist(),result['scores'].tolist()):
                    label=str(label)
                    scores[label]=max(scores.get(label,0),score)
                saved.write_text(json.dumps(scores),encoding='utf-8')
            score=scores.get(str(category),0.)
            if category==18 and score>0 and score>=threshold and getattr(self,'verify_dogs',False):
                if self.dog_verifier is None:
                    from dog_verifier import DogVerifier
                    self.dog_verifier=DogVerifier(self.device)
                score=self.dog_verifier.score(path,self,cache,threshold,score)
            matches.append(dict(file_id=key,name=item['name'],kind='image',start=None,end=None,score=score))
        matches.sort(key=lambda scene:scene['score'],reverse=True)
        return matches

    def candidates(self,items,query,threshold,cache):
        return [row for row in self.evidence(items,query,threshold,cache) if row['score']>0 and row['score']>=threshold]

    def search(self,items,query,threshold,cache):
        from library import file_results
        matches=self.candidates(items,query,threshold,cache)
        return dict(scenes=matches[:50],total=len(matches),threshold=threshold,
                    notice='物体检测候选 · 分数为检测置信度，与画面相似度不同；仍可能误检或漏检。',
                    **file_results(matches,items))
