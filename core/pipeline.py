import time
import torch
import numpy as np

from .anomaly_gate import RobustAnomalyGate
from .astrometric_centroid import AstrometricCentroidExpert
from .gpu_folding import gpu_fast_fold
from .model import AstroNetHQ
from .characterization import characterize_discovered_planet
from .gpu_peeling_search import GPUPeelingSearchEngine
from .fast_analytic_solver import fast_gpu_analytic_solver
from .micro_atmosphere_engine import MicrosecondAtmosphereEngine

class OSTE_MoE_Pipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gate = RobustAnomalyGate()
        self.centroid_expert = AstrometricCentroidExpert()
        self.peeling_engine = GPUPeelingSearchEngine(device=self.device)
        self.micro_atmo_engine = MicrosecondAtmosphereEngine(device=self.device)

        import os
        w_path = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
        self.model = AstroNetHQ().to(self.device).half()
        if os.path.exists(w_path):
            self.model.load_state_dict(torch.load(w_path, map_location=self.device))
        self.model.eval()

        # EKSİKSİZ CUDA KERNEL ISINMASI (Tüm searchsorted ve Tensor Core gecikmelerini sıfırlar)
        if self.device == "cuda":
            dummy_g = torch.randn(256, 1, 201, device=self.device).half()
            dummy_l = torch.randn(256, 1, 61, device=self.device).half()
            dummy_t = torch.linspace(0, 27.4, 6000, device=self.device)
            dummy_f = torch.ones(6000, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_g, dummy_l)
                _ = gpu_fast_fold(dummy_t, dummy_f, 3.0, 1.0, 0.1, device=self.device)
            torch.cuda.synchronize()

    def process_candidate(self, time_arr, flux_arr, period, t0, duration, img_oot=None, img_in=None, star_params=None):
        t_start = time.perf_counter()
        
        gate_res = self.gate.inspect(flux_arr)
        if not gate_res["has_anomaly"]:
            return {
                "decision": "NON_PLANET",
                "reason": "GATE_NO_ANOMALY",
                "confidence": 0.0,
                "latency_ms": (time.perf_counter() - t_start) * 1000.0
            }

        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)
        g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, period, t0, duration, device=self.device)

        dummy_batch_g = g.half().repeat(256, 1, 1)
        dummy_batch_l = l.half().repeat(256, 1, 1)
        with torch.no_grad():
            prob_ai = torch.sigmoid(self.model(dummy_batch_g, dummy_batch_l))[0].item()

        r_s = star_params.get("r_s", 1.0) if star_params else 1.0
        m_s = star_params.get("m_s", 1.0) if star_params else 1.0
        teff = star_params.get("teff", 5778.0) if star_params else 5778.0

        phys_fast = fast_gpu_analytic_solver(period, d_meas, duration, r_star=r_s, m_star=m_s, teff=teff, device=self.device)
        d_final = d_meas

        if d_final < 0.00018 or prob_ai < 0.20:
            return {
                "decision": "NON_PLANET",
                "reason": "NO_SIGNIFICANT_TRANSIT",
                "measured_depth": d_final,
                "confidence": prob_ai,
                "latency_ms": (time.perf_counter() - t_start) * 1000.0
            }

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

        if d_final >= 0.028:
            final_cls = "BINARY"
        elif prob_ai >= 0.35 and 0.00018 <= d_final < 0.028 and passed_astrometry:
            final_cls = "PLANET"
        else:
            final_cls = "NON_PLANET"

        total_lat = (time.perf_counter() - t_start) * 1000.0
        res = {
            "decision": final_cls,
            "cnn_confidence": prob_ai,
            "measured_depth": d_final,
            "impact_b": phys_fast["Impact_b"],
            "passed_astrometry": passed_astrometry,
            "latency_ms": total_lat
        }

        if final_cls == "PLANET":
            phys_full = characterize_discovered_planet(
                period, d_final, duration, impact_b=phys_fast["Impact_b"],
                r_star_solar=r_s, m_star_solar=m_s, t_star_k=teff
            )
            L_star = (r_s**2) * ((teff / 5778.0)**4)
            insol = L_star / (phys_full["SemiMajorAxis_AU"]**2)
            atmo = self.micro_atmo_engine.evaluate_atmosphere(
                phys_full["Radius_Earth"], phys_full["Mass_Earth"], phys_full["T_eq_K"], insol
            )
            res["physical_properties"] = phys_full
            res["atmosphere"] = atmo

        return res

    def discover_and_characterize_system(self, time_arr, flux_arr, img_oot=None, img_in=None, star_params=None):
        t_start = time.perf_counter()
        signals = self.peeling_engine.search_all_planets(time_arr, flux_arr, max_planets=4)
        
        system_planets = []
        r_s = star_params.get("r_s", 1.0) if star_params else 1.0
        m_s = star_params.get("m_s", 1.0) if star_params else 1.0
        teff = star_params.get("teff", 5778.0) if star_params else 5778.0

        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)

        for sig in signals:
            p_found = sig["period"]
            t0_found = sig["t0"]
            dur_found = sig["duration"]

            g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, p_found, t0_found, dur_found, device=self.device)
            dummy_batch_g = g.half().repeat(256, 1, 1)
            dummy_batch_l = l.half().repeat(256, 1, 1)
            with torch.no_grad():
                prob_ai = torch.sigmoid(self.model(dummy_batch_g, dummy_batch_l))[0].item()

            is_on_target = True
            if img_oot is not None and img_in is not None:
                cen_res = self.centroid_expert.evaluate_tpf_centroid(img_oot, img_in)
                is_on_target = cen_res["is_on_target"]

            if prob_ai >= 0.35 and is_on_target and d_meas > 0.00015:
                phys = fast_gpu_analytic_solver(p_found, d_meas, dur_found, r_star=r_s, m_star=m_s, teff=teff, device=self.device)
                L_star = (r_s**2) * ((teff / 5778.0)**4)
                insol = L_star / (phys["SemiMajorAxis_AU"]**2)
                atmo = self.micro_atmo_engine.evaluate_atmosphere(
                    phys["Radius_Earth"], phys["Mass_Earth"], phys["T_eq_K"], insol
                )

                system_planets.append({
                    "period_days": p_found,
                    "depth_ppm": d_meas * 1e6,
                    "cnn_confidence": prob_ai,
                    "physics": phys,
                    "atmosphere": atmo
                })

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "total_planets_found": len(system_planets),
            "planets": system_planets,
            "total_latency_ms": total_lat_ms
        }

OSTE_MoE_AutonomousDiscoveryPipeline = OSTE_MoE_Pipeline
