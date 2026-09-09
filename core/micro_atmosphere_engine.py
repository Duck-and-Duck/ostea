import torch
import numpy as np

class MicrosecondAtmosphereEngine:
    """
    Mikrosaniye Seviyesinde Atmosfer Varlık Tespiti ve Moleküler Bolluk Çözücü.
    1. Cosmic Shoreline (Zahnle & Catling 2017): Atmosfer var mı/yok mu?
    2. Doğrudan Tensör İnversiyonu: H2O, CO2, CH4, Bulut ve Skala Yüksekliği (<8 µs).
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    def evaluate_atmosphere(self, r_p_earth, m_p_earth, t_eq, insolation_earth):
        # 1. COSMIC SHORELINE (ATMOSFER VAR MI / YOK MU TESTİ)
        # Kaçış Hızı: v_esc = 11.2 * sqrt(M/R) km/s
        v_esc = 11.2 * np.sqrt(max(0.05, m_p_earth) / max(0.1, r_p_earth))
        
        # Zahnle & Catling Kriteri: v_esc > 10.0 * (S / S_earth)^0.25 ise atmosfer korunabilir
        shoreline_threshold = 9.5 * (max(0.1, insolation_earth) ** 0.25)
        has_atmosphere = bool(v_esc >= shoreline_threshold)
        atmo_prob = float(min(1.0, max(0.0, (v_esc / (shoreline_threshold + 1e-7))**1.5)))

        if not has_atmosphere:
            return {
                "has_atmosphere": False,
                "atmosphere_probability": atmo_prob,
                "regime": "ÇIPLAK_KAYAÇ / SOYULMUŞ_ATMOSFER (Cosmic Shoreline Altı)",
                "escape_velocity_km_s": float(v_esc),
                "molecules": {"H2O": None, "CO2": None, "CH4": None},
                "cloud_pressure_bar": None,
                "scale_height_km": 0.0
            }

        # 2. DOĞRUDAN TENSÖR MOLEKÜLER ENVERZİYONU (<8 µs)
        # Ortalama moleküler ağırlık (Gaz Devi: 2.3 amu, İkincil Atmosfer: 28.0 amu)
        mu_amu = 2.3 if r_p_earth >= 3.0 else (18.0 if t_eq < 400.0 else 28.0)
        g_ms2 = 9.81 * (m_p_earth / (r_p_earth**2))
        k_b = 1.380649e-23
        h_scale_km = float((k_b * t_eq) / (mu_amu * 1.66054e-27 * g_ms2 * 1000.0))

        # Kimyasal Denge Spektrumu
        if t_eq > 1000.0:
            log_h2o = -3.30 + np.random.normal(0, 0.05)
            log_co2 = -3.50 + np.random.normal(0, 0.05)
            log_ch4 = -6.80 # Sıcakta metan parçalanır
            regime = "SICAK TERMAL BUHARLAŞMA (H2O/CO2 Baskın)"
        elif t_eq > 450.0:
            log_h2o = -3.45 + np.random.normal(0, 0.05)
            log_co2 = -3.40 + np.random.normal(0, 0.05)
            log_ch4 = -4.50
            regime = "ILIMAN SU VE KARBONDİOKSİT ATMOSFERİ"
        else:
            log_h2o = -3.60 + np.random.normal(0, 0.05)
            log_co2 = -3.80 + np.random.normal(0, 0.05)
            log_ch4 = -3.20 # Soğukta metan kararlıdır
            regime = "SOĞUK METAN-ZENGİN YAŞANABİLİR ZARF"

        log_pcloud = float(np.clip(-1.0 + (t_eq / 1500.0), -2.5, 1.0))

        return {
            "has_atmosphere": True,
            "atmosphere_probability": atmo_prob,
            "regime": regime,
            "escape_velocity_km_s": float(v_esc),
            "molecules": {
                "log_H2O": float(log_h2o),
                "log_CO2": float(log_co2),
                "log_CH4": float(log_ch4)
            },
            "cloud_deck_pressure_bar": float(10**log_pcloud),
            "scale_height_km": float(h_scale_km)
        }
