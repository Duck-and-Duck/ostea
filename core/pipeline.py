import time
import torch
import numpy as np

from .anomaly_gate import RobustAnomalyGate
from .astrometric_centroid import AstrometricCentroidExpert
from .gpu_folding import gpu_fast_fold
from .model import AstroNetHQ
from .atmosphere_expert import AtmosphericInversionExpert
from .characterization import characterize_discovered_planet

class OSTE_MoE_Pipeline:
    """
    OSTE-MoE Master Hierarchical Pipeline.
    Katman 1'den Katman 4'e kadar tüm modülleri tek bir akışta birleştirir:
    1D Işık Eğrisi + 2D TPF Pikselleri -> Doğrulama -> Karakterizasyon -> Atmosfer
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gate = RobustAnomalyGate()
        self.centroid_expert = AstrometricCentroidExpert()
        self.atmosphere_expert = AtmosphericInversionExpert(device=self.device)

        # Katman 3 AstroNet-HQ Ağırlıklarını Yükle
        import os
        w_path = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
        self.model = AstroNetHQ().to(self.device).half()
        if os.path.exists(w_path):
            self.model.load_state_dict(torch.load(w_path, map_location=self.device))
        self.model.eval()

    def process_candidate(self, time_arr, flux_arr, period, t0, duration, img_oot=None, img_in=None, star_params=None):
        t_start = time.perf_counter()
        
        # 1. KATMAN 1: ANOMALY GATE (ERKEN ÇIKIŞ)
        gate_res = self.gate.inspect(flux_arr)
        if not gate_res["has_anomaly"]:
            return {
                "decision": "NON_PLANET",
                "reason": "GATE_NO_ANOMALY",
                "confidence": 0.0,
                "latency_ms": (time.perf_counter() - t_start) * 1000.0
            }

        # 2. KATMAN 1.5: 2D ASTROMETRİK CENTROID DOĞRULAMASI
        passed_astrometry = True
        centroid_rep = None
        if img_oot is not None and img_in is not None:
            centroid_rep = self.centroid_expert.evaluate_tpf_centroid(img_oot, img_in)
            passed_astrometry = centroid_rep["is_on_target"]
            if not passed_astrometry:
                return {
                    "decision": "BINARY",
                    "reason": "BEB_CENTROID_OFFSET",
                    "centroid_offset_arcsec": centroid_rep["offset_arcsec"],
                    "latency_ms": (time.perf_counter() - t_start) * 1000.0
                }

        # 3. KATMAN 2: GPU PREFIX-SUM KUTU KATLAMA
        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)
        g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, period, t0, duration, device=self.device)

        # 4. KATMAN 3: ASTRONET-HQ 1D-CNN VETTING
        dummy_batch_g = g.half().repeat(256, 1, 1)
        dummy_batch_l = l.half().repeat(256, 1, 1)
        with torch.no_grad():
            prob_ai = torch.sigmoid(self.model(dummy_batch_g, dummy_batch_l))[0].item()

        # Astrofiziksel Vetting Kuralları
        if d_meas >= 0.028:
            final_cls = "BINARY"
        elif prob_ai >= 0.35 and 0.00020 <= d_meas < 0.025 and passed_astrometry:
            final_cls = "PLANET"
        else:
            final_cls = "NON_PLANET"

        total_lat = (time.perf_counter() - t_start) * 1000.0

        res = {
            "decision": final_cls,
            "cnn_confidence": prob_ai,
            "measured_depth": d_meas,
            "passed_astrometry": passed_astrometry,
            "latency_ms": total_lat
        }

        # 5. KATMAN 4: GEZEGEN ONAYLANDIYSA FİZİKSEL KARAKTERİZASYON VE ATMOSFER ÇIKARIMI
        if final_cls == "PLANET":
            r_s = star_params.get("r_s", 1.0) if star_params else 1.0
            m_s = star_params.get("m_s", 1.0) if star_params else 1.0
            teff = star_params.get("teff", 5778.0) if star_params else 5778.0
            
            phys = characterize_discovered_planet(period, d_meas, duration, r_s, m_s, teff)
            atmo = self.atmosphere_expert.retrieve_atmosphere(d_meas)
            res["physical_properties"] = phys
            res["atmosphere"] = atmo

        return res
