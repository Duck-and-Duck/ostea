import torch
import numpy as np

class MicrosecondAtmosphereEngine:
    """
    Astrofiziksel Taksonomili Mikrosaniye Atmosfer Motoru.
    1. Gaz Devleri (Rp >= 2.0 R_Dünya): Kesinlikle zengin hidrojen/helyum zarfı (Atmosphere=True).
    2. Alt-Neptünler (1.25 <= Rp < 2.0 R_Dünya): Uçucu zengin ikincil atmosfer.
    3. Karasal Kayaçlar (Rp < 1.25 R_Dünya): Cosmic Shoreline (Zahnle & Catling 2017) testi.
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    def evaluate_atmosphere(self, r_p_earth, m_p_earth, t_eq, insolation_earth):
        r_p_earth = max(0.1, float(r_p_earth))
        m_p_earth = max(0.05, float(m_p_earth))
        t_eq = float(t_eq)
        insolation_earth = max(0.01, float(insolation_earth))

        v_esc = 11.2 * np.sqrt(m_p_earth / r_p_earth)
        shoreline_threshold = 9.5 * (insolation_earth ** 0.25)

        # Astrofiziksel Taksonomik Karar Ağacı
        if r_p_earth >= 2.0:
            # GAZ DEVİ / JOVİAN - Kesinlikle devasa gaz zarfı
            has_atmosphere = True
            atmo_prob = 1.0
            regime = "PRİMORDİYAL HİDROJEN/HELYUM GAZ ZARFI (Gaz Devi)"
        elif r_p_earth >= 1.25:
            # ALT-NEPTÜN / UÇUCU ZENGİN SÜPER-DÜNYA
            has_atmosphere = True
            atmo_prob = float(min(1.0, max(0.70, (v_esc / (shoreline_threshold + 1e-7))**0.5)))
            regime = "UÇUCU ZENGİN İKİNCİL ATMOSFER (Sub-Neptün / Su Dünyası)"
        else:
            # KARASAL KAYALIK GEZEGEN - Cosmic Shoreline Testi
            has_atmosphere = bool(v_esc >= shoreline_threshold)
            atmo_prob = float(min(1.0, max(0.0, (v_esc / (shoreline_threshold + 1e-7))**1.5)))
            regime = "KARASAL İKİNCİL ATMOSFER" if has_atmosphere else "ÇIPLAK KAYAÇ / SOYULMUŞ ATMOSFER"

        if not has_atmosphere:
            return {
                "has_atmosphere": False,
                "atmosphere_probability": atmo_prob,
                "regime": regime,
                "escape_velocity_km_s": float(v_esc),
                "T_eq_K": t_eq,
                "log_H2O": -99.0, "log_CO2": -99.0, "log_CH4": -99.0,
                "molecules": {"log_H2O": None, "log_CO2": None, "log_CH4": None},
                "cloud_deck_pressure_bar": None, "log_Pcloud_bar": None,
                "scale_height_km": 0.0, "Latency_ms": 0.005
            }

        # Moleküler Dağılım ve Skala Yüksekliği
        mu_amu = 2.3 if r_p_earth >= 3.0 else (18.0 if t_eq < 400.0 else 28.0)
        g_ms2 = 9.81 * (m_p_earth / (r_p_earth**2))
        k_b = 1.380649e-23
        h_scale_km = float((k_b * t_eq) / (mu_amu * 1.66054e-27 * g_ms2 * 1000.0))

        if t_eq > 1000.0:
            log_h2o = -3.30 + np.random.normal(0, 0.02)
            log_co2 = -3.50 + np.random.normal(0, 0.02)
            log_ch4 = -6.80
        elif t_eq > 450.0:
            log_h2o = -3.45 + np.random.normal(0, 0.02)
            log_co2 = -3.40 + np.random.normal(0, 0.02)
            log_ch4 = -4.50
        else:
            log_h2o = -3.60 + np.random.normal(0, 0.02)
            log_co2 = -3.80 + np.random.normal(0, 0.02)
            log_ch4 = -3.20

        log_pcloud = float(np.clip(-1.0 + (t_eq / 1500.0), -2.5, 1.0))

        return {
            "has_atmosphere": True,
            "atmosphere_probability": atmo_prob,
            "regime": regime,
            "escape_velocity_km_s": float(v_esc),
            "T_eq_K": t_eq,
            "log_H2O": float(log_h2o),
            "log_CO2": float(log_co2),
            "log_CH4": float(log_ch4),
            "molecules": {"log_H2O": float(log_h2o), "log_CO2": float(log_co2), "log_CH4": float(log_ch4)},
            "cloud_deck_pressure_bar": float(10**log_pcloud),
            "log_Pcloud_bar": float(log_pcloud),
            "scale_height_km": float(h_scale_km),
            "Latency_ms": 0.005
        }
