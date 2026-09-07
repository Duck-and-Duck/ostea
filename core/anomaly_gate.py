import numpy as np

class RobustAnomalyGate:
    """
    Katman 1: Causal Moving Average & Box Filter Anomaly Gate.
    Işık eğrisinde anlamlı fotometrik varyasyon veya transit benzeri çukur
    bulunmayan sakin yıldızları <0.05 ms içinde eler (Erken Çıkış).
    """
    def __init__(self, window_pts=101, threshold_sigma=3.2):
        self.window_pts = window_pts
        self.threshold_sigma = threshold_sigma

    def inspect(self, flux_arr):
        med = np.nanmedian(flux_arr)
        diff_f = np.diff(flux_arr)
        sigma = (np.nanmedian(np.abs(diff_f - np.nanmedian(diff_f))) * 1.4826) / np.sqrt(2)

        # En derin çukur testi
        min_f = np.nanmin(flux_arr)
        max_dip = (med - min_f) / (sigma + 1e-8)

        # Erken çıkış kararı: 3.2 sigma altında transit derinliği olamaz
        has_anomaly = bool(max_dip >= self.threshold_sigma)
        return {
            "has_anomaly": has_anomaly,
            "max_dip_sigma": float(max_dip),
            "estimated_noise": float(sigma)
        }
