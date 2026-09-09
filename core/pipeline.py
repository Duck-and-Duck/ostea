import time
import torch
import numpy as np

from .gpu_peeling_search import GPUPeelingSearchEngine
from .fast_analytic_solver import fast_gpu_analytic_solver
from .micro_atmosphere_engine import MicrosecondAtmosphereEngine
from .astrometric_centroid import AstrometricCentroidExpert
from .gpu_folding import gpu_fast_fold
from .model import AstroNetHQ

class OSTE_MoE_AutonomousDiscoveryPipeline:
    """
    Sıfır Ön Bilgili Çoklu Gezegen Otonom Keşif ve Mikrosaniye Karakterizasyon Motoru.
    1. Ham Işık Eğrisinde cuBLS Peeling ile tüm gezegenleri bulur (Kaçırma Yok).
    2. Katman 1.5 ile 2D Astrometriyi denetler.
    3. Katman 3 AstroNet-HQ ile onaylar.
    4. Fast Analytic Solver ile yörünge ve kütleyi mikrosaniyede çözer.
    5. Micro Atmosphere Engine ile atmosfer varlığı ve kimyasını çıkarır.
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.peeling_engine = GPUPeelingSearchEngine(device=self.device)
        self.atmosphere_engine = MicrosecondAtmosphereEngine(device=self.device)
        self.centroid_expert = AstrometricCentroidExpert()

        import os
        w_path = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
        self.model = AstroNetHQ().to(self.device).half()
        if os.path.exists(w_path):
            self.model.load_state_dict(torch.load(w_path, map_location=self.device))
        self.model.eval()

    def discover_and_characterize_system(self, time_arr, flux_arr, img_oot=None, img_in=None, star_params=None):
        t_start = time.perf_counter()
        
        # 1. SIFIR ÖN BİLGİ İLE TÜM GEZEGENLERİ BUL (PEELING)
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

            # 2. KATMAN 2 & 3: KATLAMA VE AI VETTING
            g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, p_found, t0_found, dur_found, device=self.device)
            dummy_g = g.half().repeat(256, 1, 1)
            dummy_l = l.half().repeat(256, 1, 1)
            with torch.no_grad():
                prob_ai = torch.sigmoid(self.model(dummy_g, dummy_l))[0].item()

            # 3. KATMAN 1.5: 2D ASTROMETRİ KONTROLÜ
            is_on_target = True
            if img_oot is not None and img_in is not None:
                cen_res = self.centroid_expert.evaluate_tpf_centroid(img_oot, img_in)
                is_on_target = cen_res["is_on_target"]

            if prob_ai >= 0.35 and is_on_target and d_meas > 0.00015:
                # 4. MİKROSANİYE ANALİTİK ÇÖZÜCÜ (Rp, a, i, b, rho)
                phys = fast_gpu_analytic_solver(p_found, d_meas, dur_found, r_star=r_s, m_star=m_s, teff=teff)
                
                # 5. MİKROSANİYE ATMOSFER VARLIK VE ELEMENT İNVERSİYONU
                L_star = (r_s**2) * ((teff / 5778.0)**4)
                insol = L_star / (phys["SemiMajorAxis_AU"]**2)
                atmo = self.atmosphere_engine.evaluate_atmosphere(
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
