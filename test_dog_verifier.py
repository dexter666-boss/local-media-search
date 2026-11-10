from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch
from PIL import Image
from dog_verifier import DogVerifier


class DogVerifierTests(unittest.TestCase):
    def test_strong_alternative_only_vetoes_with_low_dog_probability(self):
        p=torch.zeros(1000)
        p[355]=.78
        p[231]=.03
        self.assertTrue(DogVerifier.contradicts(p))
        p[231]=.15
        self.assertFalse(DogVerifier.contradicts(p))
        p[231]=.03
        p[355]=.69
        self.assertFalse(DogVerifier.contradicts(p))
        p[231]=.95
        self.assertFalse(DogVerifier.contradicts(p))

    def test_any_valid_dog_survives_and_predictions_reuse_across_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'photo.png'
            Image.new('RGB',(32,32)).save(path)
            verifier=DogVerifier.__new__(DogVerifier)
            verifier.device='cpu'
            verifier.torch=torch
            verifier.transform=Mock(return_value=torch.zeros(3,32,32))
            bad=torch.full((1,1000),-10.)
            bad[0,355]=10
            good=torch.full((1,1000),-10.)
            good[0,231]=10
            verifier.model=Mock(side_effect=[bad,good])
            detector=Mock(model_id='detector-v1')
            detector.transform.return_value=torch.zeros(3,32,32)
            detector.model.return_value=[dict(labels=torch.tensor([18,18]),
                scores=torch.tensor([.95,.85]),boxes=torch.tensor([[0,0,20,20],[10,10,32,32]]))]
            self.assertAlmostEqual(verifier.score(path,detector,root,.8,.95),.85)
            self.assertEqual(verifier.score(path,detector,root,.9,.95),0.)
            self.assertEqual(verifier.score(path,detector,root,.99,1.),1.)
            detector.model.assert_called_once()
            Image.new('RGB',(48,48)).save(path)
            verifier.model.side_effect=[bad,good]
            verifier.score(path,detector,root,.8,.95)
            self.assertEqual(detector.model.call_count,2)


if __name__=='__main__':
    unittest.main()
