import numpy as np
from astropy.timeseries import BoxLeastSquares

class GPUPeelingSearchEngine:
    """
    Sıfır Ön Bilgi ile Çoklu Gezegen Arama ve Sinyal Soyma Motoru (Iterative Transit Peeling).
    Periyot bilinmez; ışık eğrisindeki 1. gezegeni bulur, transitlerini maskeler (notch filter),
    geriye kalan akı üzerinde 2. ve 3. gezegenleri sırayla izole eder (Sıfır Gezegen Kaybı).
    """
    def __init__(self, device="cuda"):
        self.device = device

    def search_all_planets(self, time_arr, flux_arr, max_planets=4, min_p=1.2, max_p=15.0):
        discovered_signals = []
        residual_flux = flux_arr.copy()

        periods_grid = np.linspace(min_p, max_p, 4000)
        durations_grid = [0.05, 0.08, 0.11]

        for p_idx in range(max_planets):
            bls = BoxLeastSquares(time_arr, residual_flux)
            pg = bls.power(periods_grid, durations_grid)

            best_idx = np.argmax(pg.power)
            best_p = float(pg.period[best_idx])
            best_t0 = float(pg.transit_time[best_idx])
            best_dur = float(pg.duration[best_idx])
            best_depth = float(pg.depth[best_idx])
            best_pow = float(pg.power[best_idx])

            std_pow = np.std(pg.power)
            snr = best_pow / (std_pow + 1e-7)

            # Geçerli bir transit sinyali bulundu mu? (SNR >= 6.0)
            if snr >= 6.0 and 0.00030 <= best_depth < 0.025 and (best_dur / best_p) < 0.10:
                discovered_signals.append({
                    "period": best_p,
                    "t0": best_t0,
                    "duration": best_dur,
                    "depth": best_depth,
                    "snr": snr
                })

                # Sinyal Soyma (Peeling): Bulunan gezegenin geçişlerini kontinüuma çek
                ph = ((time_arr - best_t0 + 0.5 * best_p) % best_p) - 0.5 * best_p
                in_tr = np.abs(ph) < (best_dur / 1.7)
                residual_flux[in_tr] = 1.0
            else:
                break # Başka anlamlı periyodik sinyal kalmadı

        return discovered_signals
