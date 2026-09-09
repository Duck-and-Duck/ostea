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
from .gpu_peeling_search import GPUPeelingSearchEngine
from .fast_analytic_solver import fast_gpu_analytic_solver
from .micro_atmosphere_engine import MicrosecondAtmosphereEngine

class OSTE_MoE_Pipeline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gate = RobustAnomalyGate()
        self.centroid_expert = AstrometricCentroidExpert()
        self.atmosphere_expert = AtmosphericInversionExpert(device=self.device)
        self.peeling_engine = GPUPeelingSearchEngine(device=self.device)
        self.micro_atmo_engine = MicrosecondAtmosphereEngine(device=self.device)

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

        # 2. GPU PREFIX-SUM KATLAMA
        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)
        g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, period, t0, duration, device=self.device)

        # 3. 1D-CNN ASTRONET-HQ
        dummy_batch_g = g.half().repeat(256, 1, 1)
        dummy_batch_l = l.half().repeat(256, 1, 1)
        with torch.no_grad():
            prob_ai = torch.sigmoid(self.model(dummy_batch_g, dummy_batch_l))[0].item()

        # 4. ANALİTİK TRANSİT UYDURMA
        win_local = duration * 2.0
        phase_local = np.linspace(-win_local, win_local, 61)
        flux_local = l.cpu().numpy().flatten()
        f_scale = 1.0 - (d_meas * (flux_local - np.max(flux_local)) / (np.min(flux_local) - np.max(flux_local) + 1e-7))
        fit_res = fit_transit_parameters(phase_local, f_scale, d_meas, duration)
        d_final = fit_res["depth_fit"]

        # 5. ASTROFİZİKSEL KARAR AĞACI (SIRALAMA DÜZELTİLDİ)
        # Transit sinyali yoksa doğrudan NON_PLANET!
        if d_meas < 0.00018 or d_final < 0.00018 or prob_ai < 0.20:
            return {
                "decision": "NON_PLANET",
                "reason": "NO_SIGNIFICANT_TRANSIT",
                "measured_depth": d_final,
                "confidence": prob_ai,
                "latency_ms": (time.perf_counter() - t_start) * 1000.0
            }

        # Yalnızca geçerli transit varsa astrometriyi test et
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

        # Nihai Karar
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
            "depth_err": fit_res["depth_err"],
            "impact_b": fit_res["impact_b"],
            "passed_astrometry": passed_astrometry,
            "latency_ms": total_lat
        }

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
                phys = fast_gpu_analytic_solver(p_found, d_meas, dur_found, r_star=r_s, m_star=m_s, teff=teff)
                
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
