import time
import torch
import numpy as np

from .anomaly_gate import RobustAnomalyGate
from .astrometric_centroid import AstrometricCentroidExpert
from .gpu_folding import gpu_fast_fold
from .transit_fitter import fit_transit_parameters
from .model import AstroNetHQ
from .atmosphere_expert import AtmosphericInversionExpert
from .characterization import characterize_discovered_planet

class OSTE_MoE_Pipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gate = RobustAnomalyGate()
        self.centroid_expert = AstrometricCentroidExpert()
        self.atmosphere_expert = AtmosphericInversionExpert(device=self.device)

        import os
        w_path = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
        self.model = AstroNetHQ().to(self.device).half()
        if os.path.exists(w_path):
            self.model.load_state_dict(torch.load(w_path, map_location=self.device))
        self.model.eval()

    def process_candidate(self, time_arr, flux_arr, period, t0, duration, img_oot=None, img_in=None, star_params=None):
        t_start = time.perf_counter()
        
        # 1. ANOMALY GATE
        gate_res = self.gate.inspect(flux_arr)
        if not gate_res["has_anomaly"]:
            return {
                "decision": "NON_PLANET",
                "reason": "GATE_NO_ANOMALY",
                "confidence": 0.0,
                "latency_ms": (time.perf_counter() - t_start) * 1000.0
            }

        # 2. 2D ASTROMETRİK CENTROID
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

        # 3. GPU PREFIX-SUM KATLAMA
        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)
        g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, period, t0, duration, device=self.device)

        # 4. 1D-CNN ASTRONET-HQ
        dummy_batch_g = g.half().repeat(256, 1, 1)
        dummy_batch_l = l.half().repeat(256, 1, 1)
        with torch.no_grad():
            prob_ai = torch.sigmoid(self.model(dummy_batch_g, dummy_batch_l))[0].item()

        # 5. ANALİTİK TRANSİT UYDURMA (MANDEL-AGOL FITTER)
        win_local = duration * 2.0
        phase_local = np.linspace(-win_local, win_local, 61)
        flux_local = l.cpu().numpy().flatten()
        # Normalizasyonu geri çevirip derinlik uydurma
        f_scale = 1.0 - (d_meas * (flux_local - np.max(flux_local)) / (np.min(flux_local) - np.max(flux_local) + 1e-7))
        fit_res = fit_transit_parameters(phase_local, f_scale, d_meas, duration)

        d_final = fit_res["depth_fit"]

        if d_final >= 0.028:
            final_cls = "BINARY"
        elif prob_ai >= 0.35 and 0.00015 <= d_final < 0.025 and passed_astrometry:
            final_cls = "PLANET"
        else:
            final_cls = "NON_PLANET"

        total_lat = (time.perf_counter() - t_start) * 1000.0

        res = {
            "decision": final_cls,
            "cnn_confidence": prob_ai,
            "measured_depth": d_final,
            "depth_err": fit_res["depth_err"],
            "impact_b": fit_res["impact_b"],
            "passed_astrometry": passed_astrometry,
            "latency_ms": total_lat
        }

        # 6. TERMODİNAMİK BAĞLAŞIMLI KARAKTERİZASYON VE SBI ATMOSFER
        if final_cls == "PLANET":
            r_s = star_params.get("r_s", 1.0) if star_params else 1.0
            m_s = star_params.get("m_s", 1.0) if star_params else 1.0
            teff = star_params.get("teff", 5778.0) if star_params else 5778.0
            
            phys = characterize_discovered_planet(
                period, d_final, duration, impact_b=fit_res["impact_b"],
                r_star_solar=r_s, m_star_solar=m_s, t_star_k=teff,
                sigma_depth=fit_res["depth_err"]
            )
            atmo = self.atmosphere_expert.retrieve_atmosphere(
                d_final, gravity=phys["Surface_log_g"], teq_prior=phys["T_eq_K"]
            )
            res["physical_properties"] = phys
            res["atmosphere"] = atmo

        return res
