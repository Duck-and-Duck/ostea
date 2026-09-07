import numpy as np

class RobustAnomalyGate:
    """
    Katman 1: Causal Moving Average & Box-Filtered Anomaly Gate.
    Aşırı değer teorisi (Gumbel) gereği tekil noktalara bakmaz; transit süresine
    uygun 11 noktalı kutu filtresiyle (Box Filter) gürültüyü sqrt(11) kat bastırır.
    Sinyal içermeyen durağan yıldızları <0.05 ms içinde eler (Erken Çıkış).
    """
    def __init__(self, box_pts=11, threshold_sigma=3.5):
        self.box_pts = box_pts
        self.threshold_sigma = threshold_sigma

    def inspect(self, flux_arr):
        med = np.nanmedian(flux_arr)
        diff_f = np.diff(flux_arr)
        sigma_white = (np.nanmedian(np.abs(diff_f - np.nanmedian(diff_f))) * 1.4826) / np.sqrt(2)

        # 11 Noktalı Kutu Filtresi Konvolüsyonu (Transit Arama Çekirdeği)
        kernel = np.ones(self.box_pts) / float(self.box_pts)
        smoothed = np.convolve(flux_arr, kernel, mode='valid')

        # Kutu filtresi sonrası efektif gürültü
        sigma_box = sigma_white / np.sqrt(self.box_pts)

        # En derin U-çukuru
        min_smooth = np.nanmin(smoothed)
        max_dip_sigma = (med - min_smooth) / (sigma_box + 1e-8)

        has_anomaly = bool(max_dip_sigma >= self.threshold_sigma)
        return {
            "has_anomaly": has_anomaly,
            "max_dip_sigma": float(max_dip_sigma),
            "estimated_noise": float(sigma_white)
        }
