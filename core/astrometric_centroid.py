import numpy as np

class AstrometricCentroidExpert:
    """
    Katman 1.5: PRF Gauss Ağırlıklı Astrometrik Fark Görüntüleme Uzmanı.
    NASA SPOC standardında Gauss PRF çekirdeği ile kenar gürültüsü manivelasını
    sıfırlar ve 3.0 sigma istatistiksel belirsizlik kalkanını işletir.
    """
    def __init__(self, pixel_scale_arcsec=21.0, threshold_arcsec=7.5, sigma_prf=1.2, threshold_sigma=3.0):
        self.pixel_scale = pixel_scale_arcsec
        self.threshold_arcsec = threshold_arcsec
        self.sigma_prf = sigma_prf
        self.threshold_sigma = threshold_sigma

    def evaluate_tpf_centroid(self, img_oot, img_in, target_pos=(5.0, 5.0)):
        diff_raw = img_oot - img_in
        r0, c0 = target_pos
        grid_r, grid_c = np.indices(diff_raw.shape)
        dist_sq = (grid_r - r0)**2 + (grid_c - c0)**2

        # 1. Gauss PRF Ağırlık Maskesi (Kenar gürültüsünü üstel olarak yok eder)
        prf_weight = np.exp(-dist_sq / (2.0 * (self.sigma_prf * 1.8)**2))
        diff_masked = np.maximum(0.0, diff_raw) * prf_weight

        flux_sum = np.sum(diff_masked)
        if flux_sum <= 1e-7:
            return {"is_on_target": False, "offset_arcsec": 99.0, "significance": 99.0, "reason": "NO_TRANSIT_FLUX"}

        # 2. Ağırlıklı Kütle Merkezi
        centroid_r = np.sum(grid_r * diff_masked) / flux_sum
        centroid_c = np.sum(grid_c * diff_masked) / flux_sum

        offset_pixels = np.sqrt((centroid_r - r0)**2 + (centroid_c - c0)**2)
        offset_arcsec = offset_pixels * self.pixel_scale

        # 3. İstatistiki Belirsizlik (Cramer-Rao Sınırı)
        sigma_pix = max(0.05, 1.2 / np.sqrt(max(1.0, flux_sum * 50.0)))
        sigma_arcsec = sigma_pix * self.pixel_scale
        significance = offset_arcsec / sigma_arcsec

        # NASA SPOC Standardı: Mesafe küçükse VEYA sapma 3.0 sigma içindeyse ON-TARGET
        is_on_target = (offset_arcsec <= self.threshold_arcsec) or (significance <= self.threshold_sigma)

        return {
            "is_on_target": is_on_target,
            "centroid_diff": (float(centroid_r), float(centroid_c)),
            "offset_arcsec": float(offset_arcsec),
            "offset_pixels": float(offset_pixels),
            "significance_sigma": float(significance),
            "reason": "ON_TARGET" if is_on_target else "BACKGROUND_ECLIPSING_BINARY"
        }
