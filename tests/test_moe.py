import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import RobustAnomalyGate, AstrometricCentroidExpert, AstroNetHQ

class TestOSTEMoECore(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_layer1_anomaly_gate(self):
        gate = RobustAnomalyGate()
        flat_flux = np.ones(1000) + np.random.normal(0, 0.0001, 1000)
        res_quiet = gate.inspect(flat_flux)
        self.assertFalse(res_quiet["has_anomaly"])

        transit_flux = flat_flux.copy()
        transit_flux[500:520] -= 0.005
        res_tr = gate.inspect(transit_flux)
        self.assertTrue(res_tr["has_anomaly"])

    def test_layer1_5_astrometric_centroid(self):
        expert = AstrometricCentroidExpert()
        img_oot = np.ones((11, 11)) * 1000.0
        
        # On-target transit
        img_in_on = img_oot.copy()
        img_in_on[5, 5] -= 200.0
        res_on = expert.evaluate_tpf_centroid(img_oot, img_in_on)
        self.assertTrue(res_on["is_on_target"])

        # Background Eclipsing Binary (3 pixel offset)
        img_in_beb = img_oot.copy()
        img_in_beb[8, 8] -= 200.0
        res_beb = expert.evaluate_tpf_centroid(img_oot, img_in_beb)
        self.assertFalse(res_beb["is_on_target"])

    def test_layer3_astronet_forward(self):
        model = AstroNetHQ().to(self.device).half()
        dummy_g = torch.randn(2, 1, 201, device=self.device).half()
        dummy_l = torch.randn(2, 1, 61, device=self.device).half()
        with torch.no_grad():
            out = model(dummy_g, dummy_l)
        self.assertEqual(out.shape, (2, 1))

if __name__ == "__main__":
    unittest.main()
