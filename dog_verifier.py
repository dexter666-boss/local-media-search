"""Conservative ImageNet-family verification of dog detector candidates."""
import hashlib
import json
from pathlib import Path


class DogVerifier:
    # WordNet 3.0 dog.n.01 descendants in the official ImageNet1k synset index.
    # Generated and checked by evaluations/2026-11-04-open-world/animal_gate.py.
    family=tuple(range(151,269))
    model_id='convnext_tiny:IMAGENET1K_V1:wordnet-dog151-268:pad10-square:v1'

    def __init__(self,device):
        import torch
        from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
        self.torch=torch
        self.device=device
        weights=ConvNeXt_Tiny_Weights.DEFAULT
        self.transform=weights.transforms()
        self.model=convnext_tiny(weights=weights,progress=False).to(device).eval()

    @classmethod
    def contradicts(cls,probabilities):
        top=int(probabilities.argmax())
        return (top not in cls.family and float(probabilities[top])>=.70
                and float(probabilities[list(cls.family)].sum())<=.10)

    def score(self,path,detector,cache,threshold,original):
        from PIL import Image,ImageOps
        stat=path.stat()
        signature=f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{detector.model_id}|{self.model_id}'
        directory=Path(cache)/'dog-verification'
        directory.mkdir(parents=True,exist_ok=True)
        saved=directory/(hashlib.sha256(signature.encode()).hexdigest()+'.json')
        if saved.exists():
            rows=json.loads(saved.read_text(encoding='utf-8'))
        else:
            with Image.open(path) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            with self.torch.inference_mode():
                prediction=detector.model([detector.transform(image).to(self.device)])[0]
            rows=[]
            for label,score,box in zip(prediction['labels'].tolist(),prediction['scores'].tolist(),prediction['boxes'].tolist()):
                if label!=18: continue
                x1,y1,x2,y2=box
                dx,dy=(x2-x1)*.1,(y2-y1)*.1
                crop=image.crop((max(0,int(x1-dx)),max(0,int(y1-dy)),
                                 min(image.width,int(x2+dx+1)),min(image.height,int(y2+dy+1))))
                side=max(crop.size)
                square=Image.new('RGB',(side,side),(0,0,0))
                square.paste(crop,((side-crop.width)//2,(side-crop.height)//2))
                with self.torch.inference_mode():
                    probabilities=self.model(self.transform(square).unsqueeze(0).to(self.device)).softmax(-1)[0].cpu()
                rows.append(dict(score=score,veto=self.contradicts(probabilities)))
            saved.write_text(json.dumps(rows),encoding='utf-8')
        eligible=[row for row in rows if row['score']>=threshold]
        if not eligible: return original
        return max((row['score'] for row in eligible if not row['veto']),default=0.)
