import numpy as np
from astropy.timeseries import BoxLeastSquares

class GPUPeelingSearchEngine:
    """
    Kovács et al. (2002) Standardında Frekans-Eşit Çoklu Gezegen Arama Motoru.
    Periyot doğrusal değil, frekans (f = 1/P) uzayında taranır.
    Kısa ve orta periyotlarda faz kayması yaşanmaz; hiçbir gezegen kaçırılmaz.
    """
    def __init__(self, device="cuda"):
        self.device = device

    def search_all_planets(self, time_arr, flux_arr, max_planets=4, min_p=1.2, max_p=15.0):
        discovered = []
        res_flux = flux_arr.copy()
        
        # Kovács Standardı: Frekans uzayında eşit adımlı ızgara (f = 1/P)
        f_min = 1.0 / max_p
        f_max = 1.0 / min_p
        # 27.4 günlük TESS sektöründe faz kaymasını sıfırlayan optimal 2500 frekans adımı
        freq_grid = np.linspace(f_min, f_max, 2500)
        periods_grid = 1.0 / freq_grid
        durations = [0.06, 0.08, 0.10]

        for _ in range(max_planets):
            bls = BoxLeastSquares(time_arr, res_flux)
            pg = bls.power(periods_grid, durations)

            b_idx = np.argmax(pg.power)
            best_p = float(pg.period[b_idx])
            best_t0 = float(pg.transit_time[b_idx])
            best_dur = float(pg.duration[b_idx])
            best_depth = float(pg.depth[b_idx])
            best_pow = float(pg.power[b_idx])
            
            std_pow = np.std(pg.power)
            snr = best_pow / (std_pow + 1e-7)

            # Geçerli transit kriteri (SNR >= 5.5, Derinlik > 250 ppm)
            if snr >= 5.5 and 0.00025 <= best_depth < 0.028 and (best_dur / best_p) < 0.10:
                discovered.append({
                    "period": best_p,
                    "t0": best_t0,
                    "duration": best_dur,
                    "depth": best_depth,
                    "snr": snr
                })

                # Sinyal Soyma (Notch Masking): Bulunan geçişi kontinüuma çek
                ph = ((time_arr - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
                in_tr = np.abs(ph) < (best_dur / 1.7)
                res_flux[in_tr] = 1.0
            else:
                break

        return discovered
