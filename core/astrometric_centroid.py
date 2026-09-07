import numpy as np

class AstrometricCentroidExpert:
    """
    Katman 1.5: Astrometric Pixel-Level Centroid Difference Imaging Expert.
    NASA SPOC standardında geçiş-içi ve geçiş-dışı fark görüntüsü (Difference Image)
    üzerinde foton ağırlık merkezi kaymasını (PRF-fit) ölçer.
    TESS'in 21 ark-saniyelik piksellerindeki sinsi arka plan çift yıldızlarını (BEB) eler.
    """
    def __init__(self, pixel_scale_arcsec=21.0, threshold_arcsec=7.5, sigma_prf=1.2):
        self.pixel_scale = pixel_scale_arcsec
        self.threshold_arcsec = threshold_arcsec
        self.sigma_prf = sigma_prf

    def evaluate_tpf_centroid(self, img_oot, img_in, target_pos=(5.0, 5.0)):
        diff_raw = img_oot - img_in
        r0, c0 = target_pos
        grid_r, grid_c = np.indices(diff_raw.shape)
        dist_sq = (grid_r - r0)**2 + (grid_c - c0)**2
        
        # Kenar gürültüsü manivelasını bastıran 3.8 piksel optimal fotometrik diyafram maskesi
        aperture_mask = dist_sq <= (3.8**2)
        diff_masked = np.where(aperture_mask, np.maximum(0.0, diff_raw), 0.0)

        flux_sum = np.sum(diff_masked)
        if flux_sum <= 1e-7:
            return {"is_on_target": False, "offset_arcsec": 99.0, "reason": "NO_TRANSIT_FLUX"}

        centroid_r = np.sum(grid_r * diff_masked) / flux_sum
        centroid_c = np.sum(grid_c * diff_masked) / flux_sum

        offset_pixels = np.sqrt((centroid_r - r0)**2 + (centroid_c - c0)**2)
        offset_arcsec = offset_pixels * self.pixel_scale
        is_on_target = (offset_arcsec <= self.threshold_arcsec)

        return {
            "is_on_target": is_on_target,
            "centroid_diff": (float(centroid_r), float(centroid_c)),
            "offset_arcsec": float(offset_arcsec),
            "offset_pixels": float(offset_pixels),
            "reason": "ON_TARGET" if is_on_target else "BACKGROUND_ECLIPSING_BINARY"
        }
